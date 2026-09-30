#!/usr/bin/env python3
# A. 簡單題 - 測資產生器
# 產生 <group>_<case>.in / <group>_<case>.out
import os
import random
import subprocess

random.seed(20260701)

OUT_DIR = "testcases"
SOLUTION_BIN = "./A_solution"

os.makedirs(OUT_DIR, exist_ok=True)


def write_case(group, case, pairs):
    """pairs: list of (a, b) as python ints"""
    in_path = os.path.join(OUT_DIR, f"{group}_{case}.in")
    out_path = os.path.join(OUT_DIR, f"{group}_{case}.out")

    lines_in = [str(len(pairs))]
    for a, b in pairs:
        lines_in.append(f"{a} {b}")
    in_text = "\n".join(lines_in) + "\n"

    lines_out = [str(a + b) for a, b in pairs]
    out_text = "\n".join(lines_out) + "\n"

    with open(in_path, "w") as f:
        f.write(in_text)
    with open(out_path, "w") as f:
        f.write(out_text)

    return in_path, out_path


def rand_int(digits_max, allow_negative=True):
    """產生位數在 [1, digits_max] 之間的隨機整數（可正可負）"""
    d = random.randint(1, digits_max)
    if d == 1:
        val = random.randint(0, 9)
    else:
        first = random.randint(1, 9)
        rest = "".join(str(random.randint(0, 9)) for _ in range(d - 1))
        val = int(str(first) + rest)
    if allow_negative and val != 0 and random.random() < 0.5:
        val = -val
    return val


# ---------- Subtask 1：範例測資 (0 分) ----------
sample_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
write_case(1, 1, sample_pairs)


# ---------- Subtask 2：-100 <= a, b <= 100 (30 分) ----------
# 2_1：邊界值窮舉
edge_vals = [-100, -1, 0, 1, 100]
edge_pairs = [(a, b) for a in edge_vals for b in edge_vals]
write_case(2, 1, edge_pairs)

# 2_2：隨機（小樣本）
pairs = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(20)]
write_case(2, 2, pairs)

# 2_3：a = -b（互為相反數，結果為 0）
pairs = []
for _ in range(20):
    a = random.randint(-100, 100)
    pairs.append((a, -a))
write_case(2, 3, pairs)

# 2_4：隨機（大樣本，t 較大）
pairs = [(random.randint(-100, 100), random.randint(-100, 100)) for _ in range(200)]
write_case(2, 4, pairs)

# 2_5：t = 1 最小情況
pairs = [(random.randint(-100, 100), random.randint(-100, 100))]
write_case(2, 5, pairs)


# ---------- Subtask 3：|a|, |b| <= 10^100 (70 分) ----------
MAXD = 101  # 10^100 共 101 位數

# 3_1：全部為 101 位 9（最大量級），正負組合
nine101 = int("9" * 101)
pairs = [
    (nine101, nine101),
    (-nine101, -nine101),
    (nine101, -nine101),
    (-nine101, nine101),
]
write_case(3, 1, pairs)

# 3_2：剛好等於 10^100 的邊界
p10_100 = 10 ** 100
pairs = [
    (p10_100, p10_100),
    (-p10_100, -p10_100),
    (p10_100, -p10_100),
    (p10_100, 1),
    (-p10_100, -1),
]
write_case(3, 2, pairs)

# 3_3：隨機大數（位數隨機 1~101，正負隨機）
pairs = [(rand_int(MAXD), rand_int(MAXD)) for _ in range(50)]
write_case(3, 3, pairs)

# 3_4：進位測試，例如 999...9 + 1 造成整串進位
pairs = []
for d in [1, 2, 5, 10, 50, 100, 101]:
    v = int("9" * d)
    pairs.append((v, 1))
    pairs.append((-v, -1))
write_case(3, 4, pairs)

# 3_5：借位測試，結果前導零消去 / 結果為 0
pairs = []
for d in [1, 2, 5, 10, 50, 100, 101]:
    v = int("9" * d)
    pairs.append((p10_100, -(p10_100 - 1)))  # 10^100 - (10^100 - 1) = 1
    pairs.append((v, -v))  # = 0
    pairs.append((v + 1, -v))  # = 1
write_case(3, 5, pairs)

# 3_6：一大一小混合
pairs = []
for _ in range(30):
    big = rand_int(MAXD)
    small = rand_int(3)
    pairs.append((big, small))
write_case(3, 6, pairs)

# 3_7：a = -b，大数互為相反數
pairs = []
for _ in range(30):
    a = rand_int(MAXD)
    pairs.append((a, -a))
write_case(3, 7, pairs)

# 3_8：兩數皆為 0，或其中一個為 0
pairs = [(0, 0), (0, rand_int(MAXD)), (rand_int(MAXD), 0), (0, -rand_int(MAXD))]
write_case(3, 8, pairs)

# 3_9：大量測資（大 t），數值大小混合小到大
pairs = [(rand_int(MAXD), rand_int(MAXD)) for _ in range(2000)]
write_case(3, 9, pairs)

# 3_10：全部貼著上界的隨機值（101 位數為主，加大量級壓力）
def rand_101_digit(allow_negative=True):
    first = random.randint(1, 9)
    rest = "".join(str(random.randint(0, 9)) for _ in range(100))
    val = int(str(first) + rest)
    if allow_negative and random.random() < 0.5:
        val = -val
    return val

pairs = [(rand_101_digit(), rand_101_digit()) for _ in range(500)]
write_case(3, 10, pairs)


print("已產生所有測資於 ./testcases/")

# ---------- 交叉驗證：用 C++ 標準解重新計算，確認與 python 結果一致 ----------
mismatches = 0
for fname in sorted(os.listdir(OUT_DIR)):
    if not fname.endswith(".in"):
        continue
    base = fname[:-3]
    in_path = os.path.join(OUT_DIR, fname)
    out_path = os.path.join(OUT_DIR, base + ".out")

    with open(in_path, "rb") as f:
        proc = subprocess.run([SOLUTION_BIN], stdin=f, capture_output=True, text=False)
    cpp_out = proc.stdout.decode()

    with open(out_path, "r") as f:
        expected = f.read()

    if cpp_out.strip("\n") != expected.strip("\n"):
        mismatches += 1
        print(f"[MISMATCH] {base}")

if mismatches == 0:
    print("交叉驗證通過：C++ 標準解輸出與產生器計算結果完全一致。")
else:
    print(f"發現 {mismatches} 組測資不一致，請檢查！")
