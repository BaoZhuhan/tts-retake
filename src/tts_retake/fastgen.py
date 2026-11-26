"""Fast decoding for MOSS delay-pattern TTS models (MOSS-TTSD 1.0, MOSS-TTS 1.5). It runs the same per-frame loop as the models'
own generate (same masks, delay-pattern state machine and sampling rules), with three changes for speed:
- each frame is one CUDA graph replay over a static KV cache, with sampling and bookkeeping on the GPU (no Python overhead or
  host syncs inside a frame);
- continuous batching: every row of the batch is a slot with its own position, and a slot whose item finished is refilled with
  the next item at once, so a batch no longer waits for its longest item, and prompts are never padded;
- top-p runs over the top-k tokens only, and sampling is exact Gumbel-max in float32.
The audio repetition penalty is tracked per item (HF pools the whole batch)."""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

BIG = torch.iinfo(torch.int64).max


def sample(logits, k, p, greedy):
    """Top-k, then top-p over the kept tokens, then an exact sample (Gumbel-max) along the last dim."""
    if greedy:
        return logits.argmax(-1)
    vals, idx = logits.topk(min(k, logits.shape[-1]) if k and k > 0 else logits.shape[-1], -1)
    if p is not None and p < 1.0:
        probs = vals.softmax(-1)
        vals = vals.masked_fill(probs.cumsum(-1) - probs > p, float("-inf"))
    g = torch.empty_like(vals).exponential_().log_().neg_()
    return idx.gather(-1, (vals + g).argmax(-1, keepdim=True)).squeeze(-1)


class FastGen:
    def __init__(self, model, slots=8, bucket=256, check_every=8):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        self.m, self.c, self.lm = model, model.config, model.language_model
        lc = self.lm.config
        assert not getattr(lc, "use_sliding_window", False)
        self.rope, self.n_vq, self.V = apply_rotary_pos_emb, model.config.n_vq, model.config.audio_vocab_size + 1
        self.kvh, self.hd = lc.num_key_value_heads, getattr(lc, "head_dim", None) or lc.hidden_size // lc.num_attention_heads
        self.audio_w = torch.cat([h.weight for h in model.lm_heads[1:]]).detach()
        self.find = sys.modules[type(model).__module__].find_last_equal_C
        self.slots, self.bucket, self.every = slots, bucket, check_every
        self.key, self.s, self.graphs = None, None, {}

    def release(self):
        self.key, self.s, self.graphs = None, None, {}
        torch.cuda.empty_cache()

    def state(self, cap, opts):
        """Static buffers for self.slots rows of cap positions; kept while later calls fit."""
        if self.key and cap <= self.key[0] and opts == self.key[1]:
            return
        cap = max(cap, self.key[0]) if self.key else cap
        self.release()
        B, d, n = self.slots, self.audio_w.device, self.n_vq
        kv = lambda: torch.zeros(B, self.kvh, cap, self.hd, dtype=self.audio_w.dtype, device=d)
        z = lambda *shape, dtype=torch.long: torch.zeros(*shape, dtype=dtype, device=d)
        self.s = {"K": [kv() for _ in self.lm.layers], "Vv": [kv() for _ in self.lm.layers],
                  "hid": z(B, self.audio_w.shape[1], dtype=self.audio_w.dtype), "valid": z(B, cap, dtype=torch.bool),
                  "pos": z(B), "t": z(B), "cur": z(B, 1, n + 1), "out": z(B, cap, n + 1), "stop": z(B, dtype=torch.bool),
                  "is_audio": z(B, dtype=torch.bool), "alen": z(B), "dlen": z(B), "first": z(B), "seen": z(B, n, self.V, dtype=torch.bool)}
        self.s["stop"].fill_(True)
        c, Vt = self.c, self.lm.config.vocab_size
        bad0 = z(Vt, dtype=torch.bool)
        bad0[[c.pad_token_id, c.audio_assistant_gen_slot_token_id, c.audio_assistant_delay_slot_token_id, c.audio_end_token_id]] = True
        bad1 = torch.ones(Vt, dtype=torch.bool, device=d)
        bad1[[c.audio_assistant_gen_slot_token_id, c.audio_assistant_delay_slot_token_id]] = False
        first, early = z(Vt, dtype=torch.bool), z(Vt, dtype=torch.bool)
        first[151662] = True
        early[c.im_end_token_id] = True
        self.masks, self.ar, self.rows = (bad0, bad1, first, early), torch.arange(n, device=d), torch.arange(B, device=d)
        self.graphs, self.pool, self.key = {}, torch.cuda.graph_pool_handle(), (cap, opts)

    def forward(self, L):
        """One frame through the backbone, each row at its own position, attending to positions [0, L)."""
        s, B, r = self.s, self.slots, self.rows
        x = self.m.get_input_embeddings(s["cur"])
        cos, sin = self.lm.rotary_emb(x, s["pos"][:, None])
        mask = s["valid"][:, None, None, :L]
        for i, layer in enumerate(self.lm.layers):
            a, h = layer.self_attn, layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(B, 1, -1, self.hd)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(B, 1, -1, self.hd)).transpose(1, 2)
            v = a.v_proj(h).view(B, 1, -1, self.hd).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            s["K"][i][r, :, s["pos"]] = k[:, :, 0]
            s["Vv"][i][r, :, s["pos"]] = v[:, :, 0]
            o = F.scaled_dot_product_attention(q.reshape(B, self.kvh, -1, self.hd), s["K"][i][:, :, :L], s["Vv"][i][:, :, :L],
                                               attn_mask=mask, scale=a.scaling)
            x = x + a.o_proj(o.reshape(B, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        s["hid"].copy_(self.lm.norm(x)[:, 0])

    def step(self, L):
        """Sample each row's next frame from its last hidden state (the HF loop body), record it, then run it through the backbone."""
        s, c, n, (tt, tp, tk, at, ap, ak, rp) = self.s, self.c, self.n_vq, self.key[1]
        h, t, r = s["hid"], s["t"], self.rows
        stop, dlen, alen = s["stop"], s["dlen"], s["alen"]
        bad0, bad1, first, early = self.masks
        text = torch.full_like(t, c.pad_token_id)
        text = torch.where(~stop & (dlen < n), c.audio_assistant_delay_slot_token_id, text)
        eos = ~stop & (dlen == n)
        text = torch.where(eos, c.audio_end_token_id, text)
        s["is_audio"] &= ~eos
        samp = ~stop & (dlen > n)
        lt = self.m.lm_heads[0](h).float() / (tt if tt > 0 else 1.0)
        bad = torch.where(s["is_audio"][:, None], bad1, bad0) | (first & (t == 0)[:, None]) | (early & (t <= n)[:, None])
        text = torch.where(samp, sample(lt.masked_fill(bad, float("-inf")), tk, tp, tt <= 0), text)
        s["is_audio"] |= text == c.audio_start_token_id
        stop |= text == c.im_end_token_id
        la = (h @ self.audio_w.T).float().view(-1, n, self.V) / (at if at > 0 else 1.0)
        if rp != 1.0:
            la = torch.where(s["seen"], torch.where(la > 0, la / rp, la * rp), la)
        la[..., c.audio_pad_code] = float("-inf")
        la[..., -1] = float("-inf")
        post = (self.ar > (dlen[:, None] - 1)) | (dlen[:, None] == BIG)
        audio = torch.where((alen[:, None] > self.ar) & post, sample(la, ak, ap, at <= 0), c.audio_pad_code)
        if rp != 1.0:
            s["seen"].scatter_(2, audio[..., None], True)
        alen += ((text == c.audio_start_token_id) | (text == c.audio_assistant_gen_slot_token_id)
                 | (text == c.audio_assistant_delay_slot_token_id)).long()
        alen.masked_fill_(text == c.audio_end_token_id, 0)
        dlen.masked_fill_((dlen == BIG) & (text == c.audio_assistant_delay_slot_token_id), 0)
        dlen += (dlen != BIG).long()
        dlen.masked_fill_(dlen > n, BIG)
        s["first"].copy_(torch.where(stop & (s["first"] == BIG), t, s["first"]))
        s["cur"].copy_(torch.cat([text[:, None, None], audio[:, None, :]], 2))
        s["out"][r, t] = s["cur"][:, 0]
        s["valid"][r, s["pos"]] = ~stop
        self.forward(L)
        cap = s["valid"].shape[1] - 1
        t.add_(1).clamp_(max=cap)
        s["pos"].add_(1).clamp_(max=cap)

    def graph(self, L):
        if L not in self.graphs:
            keep = {k: v.clone() for k, v in self.s.items() if k not in ("K", "Vv")}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    self.step(L)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool):
                self.step(L)
            for k, v in keep.items():
                self.s[k].copy_(v)
            self.graphs[L] = g
        return self.graphs[L]

    def load(self, b, ids, rp):
        """Prefill one unpadded prompt [S, C] into slot b and reset that slot's decoding state."""
        s, c, S = self.s, self.c, ids.shape[0]
        pre = self.lm(inputs_embeds=self.m.get_input_embeddings(ids[None]), use_cache=True)
        cache = pre.past_key_values
        for i in range(len(self.lm.layers)):
            layer = cache.layers[i] if hasattr(cache, "layers") else None
            k, v = (layer.keys, layer.values) if layer is not None else (cache.key_cache[i], cache.value_cache[i])
            s["K"][i][b, :, :S] = k[0]
            s["Vv"][i][b, :, :S] = v[0]
        s["hid"][b] = pre.last_hidden_state[0, -1]
        del pre, cache
        s["valid"][b] = False
        s["valid"][b, :S] = True
        start = int(self.find(ids[None, :, 0], c.audio_start_token_id)[0])
        cont = int(ids[-1, 0]) in (c.audio_start_token_id, c.audio_assistant_gen_slot_token_id) and start != -1
        for k, v in (("pos", S), ("t", 0), ("stop", False), ("first", BIG), ("alen", S - start if cont else 0), ("is_audio", cont), ("dlen", BIG)):
            s[k][b] = v
        s["seen"][b] = False
        if rp != 1.0:
            s["seen"][b].scatter_(1, ids[:, 1:].T, True)

    @torch.inference_mode()
    def many(self, prompts, max_new_tokens, text_temperature=1.5, text_top_p=1.0, text_top_k=50, audio_temperature=1.7,
             audio_top_p=0.8, audio_top_k=25, audio_repetition_penalty=1.0):
        """Generate for each unpadded prompt [S, 1 + n_vq] (max_new_tokens: one int, or one per prompt). Returns, per prompt,
        the prompt followed by its generated frames."""
        news = [max_new_tokens] * len(prompts) if isinstance(max_new_tokens, int) else list(max_new_tokens)
        e, b = self.every, self.bucket
        self.state(-(-(max(len(p) + m for p, m in zip(prompts, news)) + e + 1) // b) * b,
                   (text_temperature, text_top_p, text_top_k, audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty))
        s, cap = self.s, self.key[0]
        queue, live, done = list(range(len(prompts))), {}, {}
        s["stop"].fill_(True)
        while queue or live:
            for slot in range(self.slots):
                if slot not in live and queue:
                    j = queue.pop(0)
                    self.load(slot, prompts[j], audio_repetition_penalty)
                    live[slot] = [j, len(prompts[j])]
            L = min(cap, -(-(max(p for _, p in live.values()) + e + 1) // b) * b)
            g = self.graph(L)
            for _ in range(e):
                g.replay()
            for v in live.values():
                v[1] += e
            first, t = s["first"].tolist(), s["t"].tolist()
            for slot in list(live):
                j = live[slot][0]
                if first[slot] != BIG or t[slot] >= news[j]:
                    steps = min(first[slot] + 1 if first[slot] != BIG else news[j], news[j])
                    done[j] = torch.cat([prompts[j], s["out"][slot, :steps]])
                    del live[slot]
                    s["stop"][slot] = True
        return [done[j] for j in range(len(prompts))]

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1000, **kw):
        """Drop-in for model.generate on a left-padded batch: same arguments (sampling defaults of MOSS-TTS 1.5) and output list."""
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool, device=input_ids.device)
        prompts = [x[m.bool()] for x, m in zip(input_ids, attention_mask)]
        out = []
        for p, g in zip(prompts, self.many(prompts, max_new_tokens, **kw)):
            i = int(self.find(p[None, :, 0], self.c.im_start_token_id)[0]) + 3
            out.append((torch.tensor(len(p) - i, device=p.device), g[i:]))
        return out
