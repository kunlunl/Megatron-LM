gbs = 64
start = 64
step = 64
iters = 100
end = start + iters * step

with open("github_subset_2.csv", "r") as f:
    lines = f.readlines()

assert lines[0] == "size\n"

seqlen = start
idx = 0
while seqlen <= end and idx + gbs <= len(lines) - 1:
    for i in range(gbs):
        lines[1 + idx + i] = str(seqlen) + '\n'
    seqlen += step
    idx += gbs

with open("github_subset_2.csv", "w") as f:
    f.writelines(lines)
