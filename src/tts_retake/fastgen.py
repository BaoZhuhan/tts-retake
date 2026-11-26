"""Fast decoding for MOSS delay-pattern TTS models (MOSS-TTSD 1.0, MOSS-TTS 1.5). It runs the same per-frame loop as the models'
own generate (same masks, delay-pattern state machine and sampling rules), but each frame is one CUDA graph replay over a static
KV cache, with sampling and bookkeeping kept on the GPU, so the step costs GPU time only, with no Python overhead or host syncs.
Differences from the HF loop: the audio repetition penalty is tracked per row (HF pools the whole batch), and sampling is exact
Gumbel-max in float32 instead of multinomial in bf16."""
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
    def __init__(self, model, bucket=512, check_every=16):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

        self.m, self.c, self.lm = model, model.config, model.language_model
        lc = self.lm.config
        assert not getattr(lc, "use_sliding_window", False)
        self.rope, self.n_vq, self.V = apply_rotary_pos_emb, model.config.n_vq, model.config.audio_vocab_size + 1
        self.kvh, self.hd = lc.num_key_value_heads, getattr(lc, "head_dim", None) or lc.hidden_size // lc.num_attention_heads
        self.audio_w = torch.cat([h.weight for h in model.lm_heads[1:]]).detach()
        self.find = sys.modules[type(model).__module__].find_last_equal_C
        self.bucket, self.every = bucket, check_every
        self.key = None

    def release(self):
        self.key, self.s, self.graphs = None, None, {}
        torch.cuda.empty_cache()

    def state(self, B, cap, opts):
        """Static buffers for B rows and cap positions; kept while later calls fit (smaller batches run padded)."""
        if self.key and B <= self.key[0] and cap <= self.key[1] and opts == self.key[2]:
            return
        if self.key:
            B, cap = max(B, self.key[0]), max(cap, self.key[1])
        self.release()
        d, L = self.audio_w.device, len(self.lm.layers)
        n = self.n_vq
        kv = lambda: torch.zeros(B, self.kvh, cap, self.hd, dtype=self.audio_w.dtype, device=d)
        self.s = {"K": [kv() for _ in range(L)], "Vv": [kv() for _ in range(L)],
                  "hid": torch.zeros(B, self.audio_w.shape[1], dtype=self.audio_w.dtype, device=d),
                  "valid": torch.zeros(B, cap, dtype=torch.bool, device=d), "pos": torch.zeros(1, dtype=torch.long, device=d),
                  "t": torch.zeros(1, dtype=torch.long, device=d), "cur": torch.zeros(B, 1, n + 1, dtype=torch.long, device=d),
                  "out": torch.zeros(B, cap, n + 1, dtype=torch.long, device=d), "stop": torch.zeros(B, dtype=torch.bool, device=d),
                  "is_audio": torch.zeros(B, dtype=torch.bool, device=d), "alen": torch.zeros(B, dtype=torch.long, device=d),
                  "dlen": torch.zeros(B, dtype=torch.long, device=d), "first": torch.zeros(B, dtype=torch.long, device=d),
                  "seen": torch.zeros(B, n, self.V, dtype=torch.bool, device=d)}
        c, Vt = self.c, self.lm.config.vocab_size
        bad0 = torch.zeros(Vt, dtype=torch.bool, device=d)
        bad0[[c.pad_token_id, c.audio_assistant_gen_slot_token_id, c.audio_assistant_delay_slot_token_id, c.audio_end_token_id]] = True
        bad1 = torch.ones(Vt, dtype=torch.bool, device=d)
        bad1[[c.audio_assistant_gen_slot_token_id, c.audio_assistant_delay_slot_token_id]] = False
        first = torch.zeros(Vt, dtype=torch.bool, device=d)
        first[151662] = True
        early = torch.zeros(Vt, dtype=torch.bool, device=d)
        early[c.im_end_token_id] = True
        self.masks = (bad0, bad1, first, early)
        self.ar = torch.arange(n, device=d)
        self.graphs, self.pool, self.key = {}, torch.cuda.graph_pool_handle(), (B, cap, opts)

    def forward(self, L):
        """One frame through the backbone at position pos, attending to positions [0, L); writes the final hidden state."""
        s, B = self.s, self.s["hid"].shape[0]
        x = self.m.get_input_embeddings(s["cur"])
        cos, sin = self.lm.rotary_emb(x, s["pos"].expand(B, 1))
        mask = s["valid"][:, None, None, :L]
        for i, layer in enumerate(self.lm.layers):
            a, h = layer.self_attn, layer.input_layernorm(x)
            q = a.q_norm(a.q_proj(h).view(B, 1, -1, self.hd)).transpose(1, 2)
            k = a.k_norm(a.k_proj(h).view(B, 1, -1, self.hd)).transpose(1, 2)
            v = a.v_proj(h).view(B, 1, -1, self.hd).transpose(1, 2)
            q, k = self.rope(q, k, cos, sin)
            s["K"][i].index_copy_(2, s["pos"], k)
            s["Vv"][i].index_copy_(2, s["pos"], v)
            o = F.scaled_dot_product_attention(q.reshape(B, self.kvh, -1, self.hd), s["K"][i][:, :, :L], s["Vv"][i][:, :, :L],
                                               attn_mask=mask, scale=a.scaling)
            x = x + a.o_proj(o.reshape(B, 1, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        s["hid"].copy_(self.lm.norm(x)[:, 0])

    def step(self, L):
        """Sample frame t from the last hidden state (the HF loop body), record it, then run it through the backbone."""
        s, c, n, (tt, tp, tk, at, ap, ak, rp) = self.s, self.c, self.n_vq, self.key[2]
        B, h, t = s["hid"].shape[0], s["hid"], s["t"]
        stop, dlen, alen = s["stop"], s["dlen"], s["alen"]
        bad0, bad1, first, early = self.masks
        text = torch.full((B,), c.pad_token_id, dtype=torch.long, device=h.device)
        text = torch.where(~stop & (dlen < n), c.audio_assistant_delay_slot_token_id, text)
        eos = ~stop & (dlen == n)
        text = torch.where(eos, c.audio_end_token_id, text)
        s["is_audio"] &= ~eos
        samp = ~stop & (dlen > n)
        lt = self.m.lm_heads[0](h).float() / (tt if tt > 0 else 1.0)
        bad = torch.where(s["is_audio"][:, None], bad1, bad0) | (first & (t == 0)) | (early & (t <= n))
        text = torch.where(samp, sample(lt.masked_fill(bad, float("-inf")), tk, tp, tt <= 0), text)
        s["is_audio"] |= text == c.audio_start_token_id
        stop |= text == c.im_end_token_id
        la = (h @ self.audio_w.T).float().view(B, n, self.V) / (at if at > 0 else 1.0)
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
        s["out"].index_copy_(1, t, s["cur"])
        s["valid"].index_copy_(1, s["pos"], (~stop)[:, None])
        self.forward(L)
        t += 1
        s["pos"] += 1

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

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=1000, text_temperature=1.5, text_top_p=1.0, text_top_k=50,
                 audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25, audio_repetition_penalty=1.0):
        """Drop-in for model.generate: same arguments (sampling defaults of MOSS-TTS 1.5) and same output list."""
        B0, S, _ = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(B0, S, dtype=torch.bool, device=input_ids.device)
        b = self.bucket
        self.state(1 << (B0 - 1).bit_length(), -(-(S + max_new_tokens + 1) // b) * b,
                   (text_temperature, text_top_p, text_top_k, audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty))
        s, c, n = self.s, self.c, self.n_vq
        B = s["hid"].shape[0]
        rows = torch.arange(B, device=input_ids.device).clamp(max=B0 - 1)
        pre = self.lm(inputs_embeds=self.m.get_input_embeddings(input_ids), attention_mask=attention_mask, use_cache=True)
        cache = pre.past_key_values
        for i in range(len(self.lm.layers)):
            layer = cache.layers[i] if hasattr(cache, "layers") else None
            k, v = (layer.keys, layer.values) if layer is not None else (cache.key_cache[i], cache.value_cache[i])
            s["K"][i][:, :, :S].copy_(k[rows])
            s["Vv"][i][:, :, :S].copy_(v[rows])
        s["hid"].copy_(pre.last_hidden_state[rows, -1])
        del pre, cache
        ids = input_ids[rows]
        s["valid"].zero_()
        s["valid"][:, :S] = attention_mask[rows].bool()
        s["pos"].fill_(S)
        s["t"].zero_()
        s["out"].zero_()
        s["stop"].zero_()
        s["first"].fill_(BIG)
        cont = (ids[:, -1, 0] == c.audio_start_token_id) | (ids[:, -1, 0] == c.audio_assistant_gen_slot_token_id)
        start = self.find(ids[..., 0], c.audio_start_token_id)
        smask = cont & (start != -1)
        s["alen"].copy_(torch.where(smask, S - start, 0))
        s["is_audio"].copy_(smask)
        s["dlen"].fill_(BIG)
        s["seen"].zero_()
        if audio_repetition_penalty != 1.0:
            s["seen"].scatter_(2, ids[..., 1:].transpose(1, 2), True)
        t = 0
        while t < max_new_tokens:
            self.graph(-(-(S + t + 1) // b) * b).replay()
            t += 1
            if t % self.every == 0 and bool((s["first"][:B0] != BIG).all()):
                break
        done = s["first"][:B0]
        steps = int(done.max()) + 1 if bool((done != BIG).all()) else max_new_tokens
        steps = min(steps, t)
        gen = torch.cat([input_ids, s["out"][:B0, :steps]], 1)
        begin = self.find(input_ids[..., 0], c.im_start_token_id) + 3
        return [(S - i, g[i:]) for i, g in zip(begin, gen)]
