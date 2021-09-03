from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import fairseq
from fairseq.modules.transformer_layer import TransformerDecoderLayer
from fairseq.modules.multihead_attention import MultiheadAttention
from fast_transformers.causal_product import causal_dot_product


class PerformerMultiheadAttention(MultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raw_permutation = self.generate_random_permutation(self.num_heads, self.kdim // self.num_heads, 0xdeadbeefdeadbeef)
        permutation = self.expand_permutation(5000, raw_permutation)
        self.register_buffer("permutation", permutation.unsqueeze(0))

    def generate_random_permutation(self, num_head, head_size, seed):
        rng = torch.Generator()
        rng.manual_seed(seed)
        permutation = [torch.randperm(head_size, generator=rng) for _ in range(num_head)]
        permutation = torch.stack(permutation, dim=0)
        return permutation

    def expand_permutation(self, max_seq_length, permutation):
        num_head, head_size = permutation.shape
        expanded = [torch.arange(head_size).unsqueeze(0).expand(num_head, head_size)]
        for _ in range(max_seq_length - 1):
            previous = expanded[-1]
            current = previous.gather(-1, permutation)
            expanded.append(current)
        expanded = torch.stack(expanded, dim=1)
        return expanded

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        assert incremental_state is None
        assert self.self_attention
        assert self.bias_k is None and self.bias_v is None
        assert not self.add_zero_attn
        assert not before_softmax
        assert not need_weights and not need_head_weights

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q = (
            q
            .view(tgt_len, bsz, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )
        k = (
            k
            .view(-1, bsz, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )
        v = (
            v
            .view(-1, bsz, self.num_heads, self.head_dim)
            .permute(1, 2, 0, 3)
        )

        q = q.gather(-1, self.permutation[:, :, :q.shape[2]].expand_as(q))
        k = k.gather(-1, self.permutation[:, :, :k.shape[2]].expand_as(k))

        q = F.relu(q) + 1e-3
        k = F.relu(k) + 1e-3

        attn = causal_dot_product(q, k, v)

        attn = attn.permute(2, 0, 1, 3).reshape(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn, None


class PerformerDecoderLayer(TransformerDecoderLayer):
    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return PerformerMultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )


fairseq.models.transformer.TransformerDecoderLayer = PerformerDecoderLayer
