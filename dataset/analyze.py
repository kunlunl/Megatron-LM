with open("github_subset_1.csv", "r") as f:
    lines = f.readlines()

assert lines[0] == "size\n"
lines = lines[1:]

sizes = []
for line in lines:
    assert line.endswith("\n")
    line = line[:-1]
    assert len(line) > 0
    sizes.append(int(line))

print(f"min: {min(sizes)}, max: {max(sizes)}, mean: {sum(sizes) / len(sizes)}")
