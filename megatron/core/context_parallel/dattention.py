# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

import functools
import pathlib
import torch

from .dispatch_flash_attn import flash_attn_func, _flash_attn_forward, _flash_attn_backward, _flash_attn_varlen_forward, _flash_attn_varlen_backward


# example before flip, rank 0: [0 1 2 3],   rank 1: [4 5 6 7],   rank 2: [8 9 10 11], rank 3: [12 13 14 15]
#          after flip, rank 0: [0 1 14 15], rank 1: [4 5 10 11], rank 2: [8 9 6 7],   rank 3: [12 13 2 3]

def flip_cp_(x, dim, world_size):
    if world_size == 1:
        return x
    batch_size = x.shape[:dim].numel()
    v = x.view(batch_size, world_size, 2, -1)[:, :, 1]

    # Fast v.copy_(v.flip(1))
    import fast_flip_cuda
    assert v.device.type == "cuda", "the fused op only supports CUDA"
    assert v.stride(2) == 1, "the fused op requires the last dim to be contiguous"
    fast_flip_cuda.flip(v.data_ptr(), v.data_ptr(), v.shape[0], v.stride(0), v.shape[1], v.stride(1), v.shape[2], v.element_size(), torch.cuda.current_stream().cuda_stream)

    return x


def flip_cp(x, dim, world_size):
    if world_size == 1:
        return x
    o = torch.empty_like(x)
    vx = x.view(*x.shape[:dim], world_size, 2, x.shape[dim] // world_size // 2, *x.shape[dim + 1:])
    vo = o.view(*x.shape[:dim], world_size, 2, x.shape[dim] // world_size // 2, *x.shape[dim + 1:])
    vo.select(dim + 1, 0).copy_(vx.select(dim + 1, 0))
    vo.select(dim + 1, 1).copy_(vx.select(dim + 1, 1).flip(dim))
    return o


@torch.compile(dynamic=True)
def mul_floordiv(x, multiplier, divisor):
    return x * multiplier // divisor


# for one unique sample_lengths instance
@functools.lru_cache(maxsize=1)
def get_sample_slice_lengths_list(sample_lengths, divisor):
    assert sample_lengths.device == torch.device("cpu"), f"Expect sample_lengths to be on cpu, got {sample_lengths.device}"
    return (sample_lengths // divisor).tolist()


# for dq0(oi0, dq1, oi1), dkv0, dkv1
@functools.lru_cache(maxsize=3)
def get_sub_sample_lengths_list(sample_lengths, multiplier, divisor):
    assert sample_lengths.device == torch.device("cpu"), f"Expect sample_lengths to be on cpu, got {sample_lengths.device}"
    return mul_floordiv(sample_lengths, multiplier, divisor).tolist()


def recover_packed_seq(x, dim, world_size, total_seq_len, packing_info=None):
    # must bypass when CP is 1
    if world_size == 1 or packing_info is None or packing_info["num_samples"] == 1:
        return x
    # put segments belonging to the same sequence together
    sample_slice_lengths_list = get_sample_slice_lengths_list(packing_info["sample_lengths"], world_size * 2)
    vo = x.view(*x.shape[:dim], -1, total_seq_len // world_size // 2, *x.shape[dim + 1:])
    o = torch.cat([sliced_seq.flatten(dim, dim + 1) for sliced_seq in vo.split(sample_slice_lengths_list, dim=dim + 1)], dim=dim)
    assert x.shape == o.shape, f"Expected output size of {x.shape}, got {o.shape}"
    return o


def slice_packed_seq(x, dim, world_size, total_seq_len, packing_info=None):
    # must bypass when CP is 1
    if world_size == 1 or packing_info is None or packing_info["num_samples"] == 1:
        return x
    sub_sample_lengths_list = get_sub_sample_lengths_list(packing_info["sample_lengths"], x.shape[dim], total_seq_len)
    vs_group = [seg.chunk(world_size * 2 * x.shape[dim] // total_seq_len, dim=dim) for seg in x.split(sub_sample_lengths_list, dim=dim)]
    o = torch.cat([slice for vs in zip(*vs_group) for slice in vs], dim=dim)
    assert x.shape == o.shape, f"Expected output size of {x.shape}, got {o.shape}"
    return o


def slice_cp(x, dim, world_size, rank, packing_info=None):
    if world_size == 1:
        assert rank == 0
        return x
    if packing_info is None:
        sample_lengths_list = [x.shape[dim]]
    else:
        sample_lengths_list = packing_info["sample_lengths"].tolist()
    vs_group = [seg.chunk(world_size * 2, dim=dim) for seg in x.split(sample_lengths_list, dim=dim)]
    return torch.cat([vs[rank * 2] for vs in vs_group] + [vs[2 * world_size - 1 - 2 * rank] for vs in vs_group], dim=dim)


def all_gather_along_dim(input, dim, group):
    world_size = torch.distributed.get_world_size(group)
    output = torch.empty(world_size, *input.shape, dtype=input.dtype, device=input.device)
    torch.distributed.all_gather_into_tensor(output, input.contiguous(), group=group)
    output = output.permute(*range(1, dim + 1), 0, *range(dim + 1, input.dim() + 1))
    output = output.reshape(*input.shape[:dim], world_size * input.shape[dim], *input.shape[dim + 1:])
    return output


def reduce_scatter_along_dim(input, dim, group):
    world_size = torch.distributed.get_world_size(group)
    output = torch.empty(input.shape[dim] // world_size, *input.shape[:dim], *input.shape[dim + 1:], dtype=input.dtype, device=input.device)
    torch.distributed.reduce_scatter_tensor(output, input.permute(dim, *range(dim), *range(dim + 1, input.dim())).contiguous(), group=group)
    output = output.permute(*range(1, dim + 1), 0, *range(dim + 1, input.dim()))
    return output


_CP_STREAM = None


def get_cp_stream():
    global _CP_STREAM
    if _CP_STREAM is None:
        _CP_STREAM = torch.cuda.Stream()
    return _CP_STREAM


# The qi refers to the i-th shard of q.
# The qi is sharded along the second axis (the seqlen axis).
# The kvT refers to kv.transpose(0, 1) whose shape is (s, b, 2, num_heads, head_dim).
# The kvTi is sharded along the first axis (the seqlen axis).
# Sharding on kvT (instead of kv) helps to avoid transpose before NCCL communication.
# Flip seqlen (call flip_cp_) before shard tensors.


class DAttentionPreFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ki, vi, cp_group):
        ctx.cp_group = cp_group
        kvTi = torch.stack([ki.transpose(0, 1), vi.transpose(0, 1)], dim=2)
        kvT = all_gather_along_dim(kvTi, 0, group=ctx.cp_group)
        flip_cp_(kvT, 0, torch.distributed.get_world_size(ctx.cp_group))
        kv = kvT.transpose(0, 1)
        return kv

    @staticmethod
    def backward(ctx, grad_kv):
        grad_kvT = grad_kv.transpose(0, 1)
        flip_cp_(grad_kvT, 0, torch.distributed.get_world_size(ctx.cp_group))
        grad_kvTi = reduce_scatter_along_dim(grad_kvT, 0, group=ctx.cp_group)
        # grad_kvTi = reduce_scatter_along_dim(grad_kvT.float(), 0, group=ctx.cp_group).to(grad_kv.dtype)
        grad_kvi = grad_kvTi.transpose(0, 1)
        return grad_kvi[:, :, 0], grad_kvi[:, :, 1], None


class DAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qi, kv, cp_group, packing_info):
        ctx.cp_group = cp_group
        ctx.packing_info = packing_info
        CP = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        b, seqlen_qi, a, d = qi.shape
        seqlen_q = seqlen_qi * CP
        seqlen_kv = kv.shape[1]

        if kv.transpose(0, 1).is_contiguous():
            data_to_save = kv.transpose(0, 1)
            ctx.convert_saved_data_to_kv = lambda x: x.transpose(0, 1)
        elif kv.permute(2, 1, 0, 3, 4).is_contiguous():
            data_to_save = kv.permute(2, 1, 0, 3, 4)
            ctx.convert_saved_data_to_kv = lambda x: x.permute(2, 1, 0, 3, 4)
        else:
            raise NotImplementedError(f"unsupported kv shape {kv.shape} stride {kv.stride()}")

        ctx.use_sdp = False
        if ctx.use_sdp:
            attn_bias = torch.full((seqlen_qi, seqlen_kv), float("-inf"), dtype=qi.dtype, device=qi.device)
            attn_bias[:seqlen_qi // 2].triu_(1 + cp_rank * seqlen_kv // CP)
            attn_bias[seqlen_qi // 2:].triu_(1 + (2 * CP - 1 - 2 * cp_rank) * seqlen_kv // CP // 2)
            attn_bias = attn_bias.expand(b, a, *attn_bias.shape)
            compute_log_sumexp = True
            dropout_p = 0.
            is_causal = False
            output, log_sumexp, philox_seed, philox_offset = \
                torch.ops.aten._scaled_dot_product_efficient_attention(
                    qi.transpose(1, 2), kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2),
                    attn_bias, compute_log_sumexp, dropout_p, is_causal)
            oi = output.transpose(1, 2)

            ctx.save_for_backward(qi, log_sumexp, attn_bias, oi, log_sumexp, philox_seed, philox_offset)
            ctx.dropout_p = dropout_p
            ctx.is_causal = is_causal
            return oi, data_to_save

        dropout_p = 0.
        softmax_scale = d ** -.5
        causal = True
        window_size = (-1, -1)
        return_softmax = False

        qi0 = qi[:, :seqlen_qi // 2]
        kv0 = kv[:, :(2 * cp_rank + 1) * seqlen_kv // (2 * CP)]

        qi1 = qi[:, seqlen_qi // 2:]
        kv1 = kv[:, :(CP - cp_rank) * seqlen_kv // CP]

        if packing_info is not None:
            if "cp_var_len_info" in packing_info:
                cu_q_lens0, max_q_len0, cu_k_lens0, max_k_len0, cu_q_lens1, max_q_len1, cu_k_lens1, max_k_len1 = packing_info["cp_var_len_info"]
            else:
                cu_seq_lens, max_seq_len, total_seq_len = [packing_info[k] for k in ["cu_seq_lens", "max_seq_len", "total_seq_len"]]
                assert total_seq_len == kv.shape[1], f"Expected total length of such seq to be {total_seq_len}, got {kv.shape[1]} instead."
                cu_q_lens0, max_q_len0, cu_q_lens1, max_q_len1 = [mul_floordiv(x, qi0.shape[1], seqlen_q) for x in [cu_seq_lens, max_seq_len]] * 2
                cu_k_lens0, max_k_len0 = [mul_floordiv(x, kv0.shape[1], kv.shape[1]) for x in [cu_seq_lens, max_seq_len]]
                cu_k_lens1, max_k_len1 = [mul_floordiv(x, kv1.shape[1], kv.shape[1]) for x in [cu_seq_lens, max_seq_len]]
                packing_info["cp_var_len_info"] = (cu_q_lens0, max_q_len0, cu_k_lens0, max_k_len0, cu_q_lens1, max_q_len1, cu_k_lens1, max_k_len1)

            if packing_info["num_samples"] > 1:
                qi01 = qi.view(-1, seqlen_qi // 2, *qi.shape[2:])
                qi0, qi1 = recover_packed_seq(qi01, 1, CP, seqlen_q, packing_info).chunk(2, dim=0)

                kv0 = recover_packed_seq(kv0, 1, CP, kv.shape[1], packing_info)

                kv1 = recover_packed_seq(kv1, 1, CP, kv.shape[1], packing_info)


        oi0, qi_padded0, k_padded0, v_padded0, out_padded0, softmax_lse0, S_dmask0, rng_state0 = (None,) * 8
        oi1, qi_padded1, k_padded1, v_padded1, out_padded1, softmax_lse1, S_dmask1, rng_state1 = (None,) * 8

        def attn_func0():
            nonlocal oi0, qi_padded0, k_padded0, v_padded0, out_padded0, softmax_lse0, S_dmask0, rng_state0
            if packing_info is None:
                oi0, qi_padded0, k_padded0, v_padded0, out_padded0, softmax_lse0, S_dmask0, rng_state0 = _flash_attn_forward(
                    qi0,
                    kv0[:, :, 0],
                    kv0[:, :, 1],
                    dropout_p,
                    softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    return_softmax=return_softmax and dropout_p > 0,
                )
            else:
                oi0, qi_padded0, k_padded0, v_padded0, out_padded0, softmax_lse0, S_dmask0, rng_state0 = _flash_attn_varlen_forward(
                        qi0.squeeze(0),
                        kv0[:, :, 0].squeeze(0),
                        kv0[:, :, 1].squeeze(0),
                        cu_q_lens0,
                        cu_k_lens0,
                        max_q_len0,
                        max_k_len0,
                        dropout_p,
                        softmax_scale,
                        causal=causal,
                        window_size=window_size,
                        return_softmax=return_softmax and dropout_p > 0
                    )
                qi_padded0, k_padded0, v_padded0 = [x.unsqueeze(0) for x in [qi_padded0, k_padded0, v_padded0]]
            assert (qi0.shape, kv0[:, :, 0].shape, kv0[:, :, 1].shape) == (qi_padded0.shape, k_padded0.shape, v_padded0.shape), "no support padding"
            assert (oi0.data_ptr(), oi0.shape, oi0.stride()) == (out_padded0.data_ptr(), out_padded0.shape, out_padded0.stride()), "no support padding"

        def attn_func1():
            nonlocal oi1, qi_padded1, k_padded1, v_padded1, out_padded1, softmax_lse1, S_dmask1, rng_state1
            if packing_info is None:
                oi1, qi_padded1, k_padded1, v_padded1, out_padded1, softmax_lse1, S_dmask1, rng_state1 = _flash_attn_forward(
                    qi1,
                    kv1[:, :, 0],
                    kv1[:, :, 1],
                    dropout_p,
                    softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    return_softmax=return_softmax and dropout_p > 0,
                )
            else:
                oi1, qi_padded1, k_padded1, v_padded1, out_padded1, softmax_lse1, S_dmask1, rng_state1 = _flash_attn_varlen_forward(
                        qi1.squeeze(0),
                        kv1[:, :, 0].squeeze(0),
                        kv1[:, :, 1].squeeze(0),
                        cu_q_lens1,
                        cu_k_lens1,
                        max_q_len1,
                        max_k_len1,
                        dropout_p,
                        softmax_scale,
                        causal=causal,
                        window_size=window_size,
                        return_softmax=return_softmax and dropout_p > 0
                    )
                qi_padded1, k_padded1, v_padded1 = [x.unsqueeze(0) for x in [qi_padded1, k_padded1, v_padded1]]
            assert (qi1.shape, kv1[:, :, 0].shape, kv1[:, :, 1].shape) == (qi_padded1.shape, k_padded1.shape, v_padded1.shape), "no support padding"
            assert (oi1.data_ptr(), oi1.shape, oi1.stride()) == (out_padded1.data_ptr(), out_padded1.shape, out_padded1.stride()), "no support padding"

        get_cp_stream().wait_stream(torch.cuda.current_stream())
        if kv0.shape[1] >= kv1.shape[1]:  # call the longer kernel first
            attn_func0()
            with torch.cuda.stream(get_cp_stream()):
                attn_func1()
        else:
            with torch.cuda.stream(get_cp_stream()):
                attn_func1()
            attn_func0()
        torch.cuda.current_stream().wait_stream(get_cp_stream())

        if packing_info is not None:
            oi0, oi1 = [x.unsqueeze(0) for x in [oi0, oi1]]
            if packing_info["num_samples"] > 1:
                oi0 = slice_packed_seq(oi0, 1, CP, seqlen_q, packing_info)
                oi1 = slice_packed_seq(oi1, 1, CP, seqlen_q, packing_info)

        oi = torch.concat([oi0, oi1], dim=1)
        ctx.save_for_backward(qi, oi, softmax_lse0, rng_state0, softmax_lse1, rng_state1)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size

        return oi, data_to_save

    @staticmethod
    def backward(ctx, grad_oi, saved_data):
        CP = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        packing_info = ctx.packing_info
        saved_data._handle.wait()
        del saved_data._handle  # break circular reference
        kv = ctx.convert_saved_data_to_kv(saved_data)

        if ctx.use_sdp:
            qi, log_sumexp, attn_bias, oi, log_sumexp, philox_seed, philox_offset = ctx.saved_tensors
            b, seqlen_qi, a, d = qi.shape
            seqlen_kv = seqlen_qi * CP

            grad_input_mask = ctx.needs_input_grad[:3] + (False,)
            grad_qi, grad_k, grad_v, grad_bias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                grad_oi.transpose(1, 2), qi.transpose(1, 2), kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2), attn_bias, oi.transpose(1, 2),
                log_sumexp, philox_seed, philox_offset, ctx.dropout_p, grad_input_mask, ctx.is_causal)
            grad_qi, grad_k, grad_v = grad_qi.transpose(1, 2), grad_k.transpose(1, 2), grad_v.transpose(1, 2)
            grad_kv = torch.empty_strided(kv.shape, kv.stride(), dtype=kv.dtype, device=kv.device)
            grad_kv[:, :, 0] = grad_k
            grad_kv[:, :, 1] = grad_v
            return grad_qi, grad_kv, None, None, None, None

        qi, oi, softmax_lse0, rng_state0, softmax_lse1, rng_state1 = ctx.saved_tensors
        out_padded0, out_padded1 = oi.chunk(2, dim=1)
        b, seqlen_qi, a, d = qi.shape
        seqlen_q = seqlen_qi * CP
        seqlen_kv = seqlen_qi * CP

        dqi = torch.empty_like(qi)
        dkv = torch.empty_strided(kv.shape, kv.stride(), dtype=kv.dtype, device=kv.device)

        qi0 = qi[:, :seqlen_qi // 2]
        kv0 = kv[:, :(2 * cp_rank + 1) * seqlen_kv // (2 * CP)]
        doi0 = grad_oi[:, :seqlen_qi // 2]
        dqi0 = dqi[:, :seqlen_qi // 2]

        qi1 = qi[:, seqlen_qi // 2:]
        kv1 = kv[:, :(CP - cp_rank) * seqlen_kv // CP]
        doi1 = grad_oi[:, seqlen_qi // 2:]
        dqi1 = dqi[:, seqlen_qi // 2:]

        kv0_is_longer = kv0.shape[1] >= kv1.shape[1]
        if kv0_is_longer:
            dkv0 = dkv[:, :kv0.shape[1]]
            dkv1 = torch.empty_like(kv1)
        else:
            dkv0 = torch.empty_like(kv0)
            dkv1 = dkv[:, :kv1.shape[1]]

        if packing_info is not None:
            cu_q_lens0, max_q_len0, cu_k_lens0, max_k_len0, cu_q_lens1, max_q_len1, cu_k_lens1, max_k_len1 = packing_info["cp_var_len_info"]

            if packing_info["num_samples"] > 1:
                qi01 = qi.view(-1, seqlen_qi // 2, *qi.shape[2:])
                doi01 = grad_oi.view(-1, seqlen_qi // 2, *grad_oi.shape[2:])
                qi0, qi1 = recover_packed_seq(qi01, 1, CP, seqlen_q, packing_info).chunk(2, dim=0)
                doi0, doi1 = recover_packed_seq(doi01, 1, CP, seqlen_q, packing_info).chunk(2, dim=0)

                kv0 = recover_packed_seq(kv0, 1, CP, kv.shape[1], packing_info)

                kv1 = recover_packed_seq(kv1, 1, CP, kv.shape[1], packing_info)

        get_cp_stream().wait_stream(torch.cuda.current_stream())
        if packing_info is None:
            _flash_attn_backward(
                doi0,
                qi0,
                kv0[:, :, 0],
                kv0[:, :, 1],
                out_padded0,
                softmax_lse0,
                dqi0,
                dkv0[:, :, 0],
                dkv0[:, :, 1],
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size,
                rng_state=rng_state0,
            )
            with torch.cuda.stream(get_cp_stream()):
                _flash_attn_backward(
                    doi1,
                    qi1,
                    kv1[:, :, 0],
                    kv1[:, :, 1],
                    out_padded1,
                    softmax_lse1,
                    dqi1,
                    dkv1[:, :, 0],
                    dkv1[:, :, 1],
                    ctx.dropout_p,
                    ctx.softmax_scale,
                    ctx.causal,
                    ctx.window_size,
                    rng_state=rng_state1,
                )
        else:
            _flash_attn_varlen_backward(
                doi0.squeeze(0),
                qi0.squeeze(0),
                kv0[:, :, 0].squeeze(0),
                kv0[:, :, 1].squeeze(0),
                out_padded0.squeeze(0),
                softmax_lse0,
                dqi0.squeeze(0),
                dkv0[:, :, 0].squeeze(0),
                dkv0[:, :, 1].squeeze(0),
                cu_q_lens0,
                cu_k_lens0,
                max_q_len0,
                max_k_len0,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                ctx.window_size,
                rng_state=rng_state0
            )
            with torch.cuda.stream(get_cp_stream()):
                _flash_attn_varlen_backward(
                    doi1.squeeze(0),
                    qi1.squeeze(0),
                    kv1[:, :, 0].squeeze(0),
                    kv1[:, :, 1].squeeze(0),
                    out_padded1.squeeze(0),
                    softmax_lse1,
                    dqi1.squeeze(0),
                    dkv1[:, :, 0].squeeze(0),
                    dkv1[:, :, 1].squeeze(0),
                    cu_q_lens1,
                    cu_k_lens1,
                    max_q_len1,
                    max_k_len1,
                    ctx.dropout_p,
                    ctx.softmax_scale,
                    ctx.causal,
                    ctx.window_size,
                    rng_state=rng_state1
                )
        torch.cuda.current_stream().wait_stream(get_cp_stream())

        if packing_info is not None and packing_info["num_samples"] > 1:
            dqi = dqi.view(-1, seqlen_qi // 2, *dqi.shape[2:])
            dqi = slice_packed_seq(dqi, 1, CP, seqlen_q, packing_info).view(-1, seqlen_qi, *dqi.shape[2:])
            if kv0_is_longer:
                dkv0.copy_(slice_packed_seq(dkv0, 1, CP, kv.shape[1], packing_info))
                dkv1 = slice_packed_seq(dkv1, 1, CP, kv.shape[1], packing_info)
            else:
                dkv0 = slice_packed_seq(dkv0, 1, CP, kv.shape[1], packing_info)
                dkv1.copy_(slice_packed_seq(dkv1, 1, CP, kv.shape[1], packing_info))

        if kv0_is_longer:
            dkv[:, :dkv1.shape[1]] += dkv1
        else:
            dkv[:, :dkv0.shape[1]] += dkv0
        dkv[:, max(dkv0.shape[1], dkv1.shape[1]):] = 0
        return dqi, dkv, None, None, None, None, None


class ShardSaveForBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, data_to_save, group):
        ctx.group = group
        ctx.world_size = torch.distributed.get_world_size(group)
        ctx.rank = torch.distributed.get_rank(group)
        assert data_to_save.is_contiguous()
        data_shard = data_to_save.view(ctx.world_size, -1)[ctx.rank].clone()
        ctx.shape = data_to_save.shape
        ctx.save_for_backward(data_shard)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        data_shard, = ctx.saved_tensors
        saved_data = torch.empty(ctx.shape, dtype=data_shard.dtype, device=data_shard.device)
        saved_data._handle = torch.distributed.all_gather_into_tensor(saved_data, data_shard, group=ctx.group, async_op=True)
        return grad_output, saved_data, None


class FlipInplaceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, CP):
        ctx.CP = CP
        # Convert the argument of flip_cp_ to continuous layout.
        # Refer to dattention_overlap for KV layout.
        kv_2sbad = kv.permute(2, 1, 0, 3, 4)
        flip_cp_(kv_2sbad, 1, CP)
        return kv

    @staticmethod
    def backward(ctx, dkv):
        dkv_2sbad = dkv.permute(2, 1, 0, 3, 4)
        flip_cp_(dkv_2sbad, 1, ctx.CP)
        return dkv, None


class ForwardGatherBackwardSliceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim, cp_group):
        CP = torch.distributed.get_world_size(cp_group)
        cp_rank = torch.distributed.get_rank(cp_group)
        ctx.dim = dim
        ctx.CP = CP
        ctx.cp_rank = cp_rank
        x = all_gather_along_dim(x, dim, cp_group)
        flip_cp_(x, dim, CP)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return slice_cp(grad_output, ctx.dim, ctx.CP, ctx.cp_rank), None, None


def dattention(qi, ki, vi, cp_group, packing_info=None):
    if torch.distributed.get_world_size(cp_group) == 1:
        return flash_attn_func(qi, ki, vi, causal=True)
    kv = DAttentionPreFunction.apply(ki, vi, cp_group)
    oi, data_to_save = DAttentionFunction.apply(qi, kv, cp_group, packing_info)
    return oi, data_to_save


def dattention_overlap(qi, kv_2sbad, cp_group, packing_info=None):
    """The layout of kv_2sbad is (2, s, b, num_heads, head_dim).
    This is the native layout after gathering V and K respectively.
    """
    CP = torch.distributed.get_world_size(cp_group)
    assert CP >= 2, "dattention overlap is not optimized for CP=1"
    kv = kv_2sbad.permute(2, 1, 0, 3, 4)
    kv = FlipInplaceFunction.apply(kv, CP)
    oi, data_to_save = DAttentionFunction.apply(qi, kv, cp_group, packing_info)
    return oi, data_to_save


def shard_save_for_backward(x, data_to_save, group):
    return ShardSaveForBackwardFunction.apply(x, data_to_save, group)


def forward_gather_backward_slice(x, dim, cp_group):
    return ForwardGatherBackwardSliceFunction.apply(x, dim, cp_group)
