import sys


if __name__ == "__main__":

    max_seqlen = int(sys.argv[1])
    profile_batch_loops = int(sys.argv[2])
    train_iters = int(sys.argv[3])
    global_batch_size = int(sys.argv[4])

    file_inp = "/workspace/hot-switch/dataset/github_subset_1.csv"
    file_out = "/workspace/hot-switch/dataset/github_subset_2.csv"

    with open(file_inp, "r") as f:
        lines = f.readlines()
    assert lines[0] == "size\n"
    seqlens = [int(line.strip()) for line in lines[1:]]
    filtered_seqlens = [seqlen for seqlen in seqlens if seqlen <= max_seqlen]
    
    out_lengths = []
    for i in range(train_iters // profile_batch_loops):
        target_batch = filtered_seqlens[i * global_batch_size : (i + 1) * global_batch_size]
        for _ in range(profile_batch_loops):
            out_lengths.extend(target_batch)
    
    with open(file_out, "w") as f:
        f.write("size\n")
        for length in out_lengths:
            f.write(str(length) + "\n")
