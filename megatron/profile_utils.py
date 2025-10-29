# Copyright (c) 2024, Kuaishou Technology. All rights reserved.

import contextlib
import functools
import torch


PERF_MODEL = False
GLOBAL_EVENTS = list()


def record_event(arg):
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    if PERF_MODEL:
        GLOBAL_EVENTS.append((arg, ev))


class WrapInputsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, msg, *args):
        ctx.msg = msg
        record_event((msg, "forward", "begin"))
        return args

    @staticmethod
    def backward(ctx, *grads):
        record_event((ctx.msg, "backward", "end"))
        return None, *grads


class WrapOutputsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, msg, *args):
        ctx.msg = msg
        record_event((msg, "forward", "end"))
        return args

    @staticmethod
    def backward(ctx, *grads):
        record_event((ctx.msg, "backward", "begin"))
        return None, *grads


def annotate_forward_backward(forward_msg, backward_msg):
    def decorator(forward_orig):
        @functools.wraps(forward_orig)
        def wrapper(*args, **kwargs):
            inputs_tuple = args + tuple(kwargs.values())
            inputs_tuple_applied = list(WrapInputsFunction.apply(forward_msg, *inputs_tuple))
            for i, (a, b) in enumerate(zip(inputs_tuple, inputs_tuple_applied)):
                if isinstance(a, torch.nn.Parameter):
                    b = torch.nn.Parameter(b, requires_grad=a.requires_grad)
                if hasattr(a, "main_grad"):
                    b.main_grad = a.main_grad
                inputs_tuple_applied[i] = b
            args = inputs_tuple_applied[:len(args)]
            kwargs = dict(zip(kwargs.keys(), inputs_tuple_applied[len(args):]))
            outputs = forward_orig(*args, **kwargs)
            if isinstance(outputs, tuple):
                outputs = WrapOutputsFunction.apply(backward_msg, *outputs)
            else:
                outputs, = WrapOutputsFunction.apply(backward_msg, outputs)
            return outputs
        return wrapper
    return decorator


@contextlib.contextmanager
def annotate_range(msg):
    record_event((msg, "forward", "begin"))
    try:
        yield
    finally:
        record_event((msg, "forward", "end"))


def annotate_forward_range(msg):
    return annotate_range(msg)


class BackwardPushFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, msg, x):
        ctx.msg = msg
        return x

    @staticmethod
    def backward(ctx, grad):
        record_event((ctx.msg, "backward", "begin"))
        return None, grad


class BackwardPopFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, msg, x):
        ctx.msg = msg
        return x

    @staticmethod
    def backward(ctx, grad):
        record_event((ctx.msg, "backward", "end"))
        return None, grad


@contextlib.contextmanager
def annotate_backward_range(msg):
    x1 = torch.empty(1, device="cuda", requires_grad=True)
    x1 = BackwardPushFunction.apply(msg, x1)
    x1.backward(x1)
    try:
        yield
    finally:
        x2 = torch.empty(1, device="cuda", requires_grad=True)
        x2 = BackwardPopFunction.apply(msg, x2)
        x2.backward(x1)


def enable_perf_model():
    global PERF_MODEL
    PERF_MODEL = True


def perf_model_summary():
    # fb: forward or backward
    # be: begin or end
    j_used = set()
    m = dict()
    rewrite_rules = [
        [("post", "forward", "begin"), -1, ("L", "forward" "end")],
        [("post", "forward", "end"), 1, ("forward", "forward" "end")],
        [("post", "backward", "begin"), -1, ("backward", "backward" "begin")],
        [("post", "backward", "end"), 1, ("L", "backward" "begin")],
    ]
    for i, ((iname, ifb, ibe), iev) in enumerate(GLOBAL_EVENTS):
        if ibe != "begin":
            continue
        exempt_from_repeat_check = False
        for j, ((jname, jfb, jbe), jev) in enumerate(GLOBAL_EVENTS[i + 1:]):
            if iname == jname and ifb == jfb and ibe == "begin" and jbe == "end":
                break
            if iname == "emb" and ifb == "backward" and ibe == "begin":
                if jname == "backward" and jfb == "backward" and jbe == "end":
                    assert GLOBAL_EVENTS[i + 1 + j - 1][0][0] == "emb"
                    exempt_from_repeat_check = True
                    break
        else:
            raise RuntimeError(f"no match event for ({iname}, {ifb}, {ibe})")
        j = i + j + 1

        if iname.startswith("L"):
            iname = "L"
        if jname.startswith("L"):
            jname = "L"
        for arg1, delta_index, arg2 in rewrite_rules:
            if arg1 == GLOBAL_EVENTS[i]:
                assert 0 <= j + delta_index < len(GLOBAL_EVENTS)
                (jname, jfb, jbe), jev = GLOBAL_EVENTS[j + delta_index]

        jev.synchronize()
        if not exempt_from_repeat_check:
            assert j not in j_used
            j_used.add(j)
        delta_t = iev.elapsed_time(jev)
        if (iname, ifb) not in m:
            m[iname, ifb] = []
        m[iname, ifb].append(delta_t)

    import megatron.core.parallel_state as mpu
    from megatron import get_args

    cp_size = get_args().context_parallel_size
    cp_group = mpu.get_context_parallel_group(cp_size)
    cp_rank = torch.distributed.get_rank(group=cp_group)

    num_layers = get_args().num_layers
    mbs = len(m["L", "backward"]) // num_layers
    assert len(m["emb", "forward"]) == mbs
    assert len(m["emb", "backward"]) == mbs
    assert len(m["post", "forward"]) == mbs
    assert len(m["post", "backward"]) == mbs
    dp_rank = mpu.get_data_parallel_rank()

    time_dict = dict()
    for (name, fb), raw_time_list in sorted(m.items()):
        for i in range(mbs):
            items_per_mbs = len(raw_time_list) // mbs
            time_list = raw_time_list[i * items_per_mbs:(i + 1) * items_per_mbs]

            if name == "L":
                line = ""
                for item in time_list:
                    line += f"{item} "
                with open(f"L_{fb}_dp{dp_rank}.txt", "a") as f:
                    f.write(line + "\n")
    
            if name == "L":
                print(f"L rank={torch.distributed.get_rank()}, {name} {fb} {time_list}")

            time_list = torch.tensor(time_list[len(time_list) // 2:], dtype=torch.float32, device="cuda").mean()
            torch.distributed.all_reduce(time_list, op=torch.distributed.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

            # TODO(hot-switch): Find the correct way to handle cp size.
            cp_size = get_args().context_parallel_size
            torch.distributed.all_reduce(time_list, op=torch.distributed.ReduceOp.SUM, group=cp_group)

            # TODO(hot-switch): Cannot do reduce on dp because seqlens are different.
            # torch.distributed.all_reduce(time_list, op=torch.distributed.ReduceOp.MAX, group=mpu.get_data_parallel_group())

            # TODO(hot-switch): Find the correct way to handle cp size.
            time_dict[name, fb, i] = time_list.item() / (mpu.get_tensor_model_parallel_world_size() * cp_size)

    # TODO(hot-switch): Find the correct way to handle cp rank.
    # if mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_data_parallel_rank() == 0:
    if cp_rank == 0 and mpu.get_tensor_model_parallel_rank() == 0:
        dp_rank = mpu.get_data_parallel_rank()
        args = get_args()
        for i in range(mbs):
            # TODO(hot-switch): Find the correct way to handle cp size.
            # line = f"{args.hidden_size} {args.ffn_hidden_size} {args.num_attention_heads} " + \
            #     f"{args.group_query_attention} {args.num_query_groups} {args.num_layers} {args.seq_length} " + \
            #     f"{args.tensor_model_parallel_size} {mpu.get_context_parallel_world_size()} {args.kaimm_overlap_cp_slow_ctas}"
            line = f"{args.hidden_size} {args.ffn_hidden_size} {args.num_attention_heads} " + \
                f"{args.group_query_attention} {args.num_query_groups} {args.num_layers} {args.seq_length} " + \
                f"{args.tensor_model_parallel_size} {cp_size} {args.kaimm_overlap_cp_slow_ctas} {i}"
            for name in ["emb", "L", "post"]:
                print(f"{name} {time_dict[name, 'forward', i]:.3f}+{time_dict[name, 'backward', i]:.3f}", end="    ")
                line += f" {name} {time_dict[name, 'forward', i]} {time_dict[name, 'backward', i]}"
            print()
            with open(f"profile_utils_log_dp{dp_rank}.txt", "a") as f:
                f.write(line + "\n")


def perf_model_clear():
    GLOBAL_EVENTS.clear()
