# coding=utf-8

# The following code has been taken from https://github.com/NVIDIA/NeMo/blob/ \
# 782b4e1652aaa43c8be390d9db0dc89544afa080/nemo/collections/nlp/modules/ \
# common/megatron/rotary_pos_embedding.py

import functools
import warnings
import importlib.util
import torch
from torch import Tensor
from megatron import get_args
from megatron.core.context_parallel import dattention
from megatron.core import mpu
from torch import einsum, nn
from typing import Optional
try:
    from apex.transformer.functional import (
        fused_apply_rotary_pos_emb_thd,
    )

    HAVE_APPLY_ROPE_FUSION = True
except:
    HAVE_APPLY_ROPE_FUSION = False
__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_theta=10000., use_fast_rope=False, context_parallel_world_size=1, context_parallel_rank=0):
        super().__init__()
        self.dim = dim
        self.use_fast_rope = use_fast_rope
        self.use_thd_rope = self.use_fast_rope and get_args().sft_concat and context_parallel_world_size == 1
        if self.use_thd_rope and self.use_fast_rope:
            warnings.warn("Using fused_apply_rotary_pos_emb_thd instead of fast_rotary_pos_emb")
        self.context_parallel_world_size = context_parallel_world_size
        self.context_parallel_rank = context_parallel_rank
        self.register_buffer('dummy_buffer', torch.tensor(1.))
        if importlib.util.find_spec('einops') is None:
            raise RuntimeError("einops is required for Rotary Embedding")
        self.forward = functools.lru_cache(maxsize=1)(self.forward)

    def forward(self, seq_len, offset=0, sample_lengths=None):
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.dim, 2, device=self.dummy_buffer.device).float() / self.dim))
        if sample_lengths is not None and not self.use_thd_rope:
            assert sum(sample_lengths) == seq_len
            seq = torch.cat([torch.arange(sample_length, device=inv_freq.device) for sample_length in sample_lengths]) + offset
        else:
            seq = torch.arange(seq_len, device=inv_freq.device) + offset
        seq = dattention.slice_cp(seq, 0, self.context_parallel_world_size, self.context_parallel_rank, sample_lengths=sample_lengths)
        freqs = einsum('i , j -> i j', seq.type_as(inv_freq), inv_freq)
        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        from einops import rearrange
        freqs = rearrange(emb, 'n d -> n 1 1 d')
        if (not self.use_thd_rope) and self.use_fast_rope:
            freqs_sin_cos = torch.cat([freqs.sin()[..., None], freqs.cos()[..., None]], dim=-1).reshape(*freqs.shape[:-1], -1)
            return freqs_sin_cos.type_as(self.dummy_buffer)
        # Note(yuantailing): Store freqs (before sin/cos) in fp32 to match fast rope precision
        return freqs


class FastRotaryPosEmbFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, freqs, precompute_sin_cos):
        import fast_rotary_pos_emb
        seq_length = t.shape[0]
        assert freqs.shape[0] == seq_length, \
            f"due to flip mechanism of context parallel impl, sequence length of freqs: {freqs.shape[0]} must match that of t: {seq_length}."
        output = fast_rotary_pos_emb.forward(t, freqs, precompute_sin_cos)
        ctx.save_for_backward(freqs)
        ctx.precompute_sin_cos = precompute_sin_cos
        return output

    @staticmethod
    def backward(ctx, grad_output):
        import fast_rotary_pos_emb
        freqs, = ctx.saved_tensors
        d_t = fast_rotary_pos_emb.backward(grad_output, freqs, ctx.precompute_sin_cos)
        return d_t, None, None


def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_bshd(t, freqs, use_fast_rope=False):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    if use_fast_rope:
        return FastRotaryPosEmbFunction.apply(t, freqs, True)
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    # Note(yuantailing): Calculate sin/cos in fp32 to match fast rope precision
    t = (t * freqs.cos().to(t.dtype)) + (_rotate_half(t) * freqs.sin().to(t.dtype))
    return torch.cat((t, t_pass), dim=-1)
def apply_rotary_pos_emb_thd(t: Tensor, cu_seqlens: Tensor, freqs: Tensor) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.
    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]
    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    splits = torch.split(t, seqlens)
    outputs = []
    for i, x in enumerate(splits):
        reshaped_x = x
        freq_slice = freqs[:reshaped_x.size(0)]
        outputs.append(apply_rotary_pos_emb_bshd(reshaped_x, freq_slice))
    return torch.cat(outputs,dim=0)


def apply_rotary_pos_emb(
    t: Tensor, freqs: Tensor, cu_seqlens: Optional[Tensor] = None, use_fast_rope=False
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd (conventional) / thd (packed seq) format
    """
    # return t.contiguous()

    if use_fast_rope:
        if cu_seqlens is None or mpu.get_context_parallel_world_size() > 1:
            return FastRotaryPosEmbFunction.apply(t, freqs, True)
        else:
            t = t.squeeze(1)
            return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs).unsqueeze(1)
    else:
        if cu_seqlens is None or mpu.get_context_parallel_world_size() > 1:
            return apply_rotary_pos_emb_bshd(t, freqs)
        else:
            return apply_rotary_pos_emb_thd(t, cu_seqlens, freqs)