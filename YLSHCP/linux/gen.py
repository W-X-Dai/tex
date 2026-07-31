#!/usr/bin/env python3
import subprocess, random, string, os

SOLUTION = "./sol"
OUTDIR   = "./testdata"
os.makedirs(OUTDIR, exist_ok=True)

def run(inp): 
    r = subprocess.run([SOLUTION], input=inp, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout

def save(group, case, lines):
    inp = "\n".join(lines) + "\n"
    base = f"{OUTDIR}/{group}_{case}"
    open(base+".in","w").write(inp)
    open(base+".out","w").write(run(inp))
    print(f"  [{group}_{case}] {len(lines)} 行")

def build_tree(structure, lines):
    for name, children in structure.items():
        if children is None: lines.append(f"touch {name}")
        else:
            lines.append(f"mkdir {name}")
            lines.append(f"cd {name}")
            build_tree(children, lines)
            lines.append("cd ..")

_C = string.ascii_lowercase + string.digits + "_"
def rname(a=2,b=5): return "".join(random.choices(_C, k=random.randint(a,b)))
def hname(): return "."+rname(1,4)

# ── S0 範例 ────────────────────────────────────────────────────────────────────
print("S0")
save(0,1,[
    "pwd",
    "mkdir foo","mkdir bar","touch baz",
    "ls",
    "cd foo","pwd",
    "mkdir sub","touch .hide","la",
    "cd ..","cd ..",
    "cd /foo/sub","pwd",
    "touch file","cd /",
    "rm baz","rm -r foo",
    "ls",
    "mv bar renamed","ls",
    "mkdir dir","mv renamed dir",
    "cd dir","ls","pwd",
    "cd /dir/renamed","pwd",
    "^C",
])

# ── S1 單層名稱 ────────────────────────────────────────────────────────────────
print("S1")
save(1,1,[
    "pwd",
    "mkdir apple","mkdir banana","mkdir cherry",
    "touch zebra","touch .env","touch .gitignore",
    "ls","la",
    "cd apple","pwd",
    "mkdir x","mkdir y","touch f1","touch f2","touch .hidden",
    "ls","la",
    "cd y","pwd","cd ..",
    "rm f1","rm f2",
    "ls","la",
    "cd ..","pwd","ls",
    "^C",
])

random.seed(42)
names  = sorted(set(rname(2,8) for _ in range(70)))[:50]
hidden = sorted(set(hname()    for _ in range(12)))[:8]
lines  = []
for n in names:  lines.append(f"mkdir {n}")
for h in hidden: lines.append(f"touch {h}")
lines += ["ls","la"]
lines.append(f"cd {names[0]}")
inner = sorted(set(rname(1,5) for _ in range(35)))[:25]
for n in inner: lines.append(f"touch {n}")
lines.append("ls")
for n in inner[:12]: lines.append(f"rm {n}")
lines.append("ls")
lines.append("cd ..")
for n in names[1:6]: lines += [f"cd {n}","pwd","cd .."]
lines.append("ls")
lines.append("^C")
save(1,2,lines)

save(1,3,[
    "ls","la",
    "mkdir a","cd a","ls","la","cd ..",
    "mkdir b","touch b_file",
    "cd b","pwd","ls","la","cd ..",
    "^C",
])

# ── S2 相對路徑 ────────────────────────────────────────────────────────────────
print("S2")
save(2,1,[
    "mkdir a",
    "cd a","mkdir b",
    "cd b","mkdir c",
    "cd c","pwd",
    "cd ..","pwd",
    "cd ../..","pwd",
    "cd ..","pwd",              # 根目錄 cd .. → 維持
    "cd a/b/c","pwd",
    "cd ../../..","pwd",
    "cd a","cd ./b","pwd",
    "cd ./../b/.","pwd",
    "touch file","rm ./file","ls",
    "cd c","mkdir d","cd d","pwd",   # cur 此時在 /a/b，cd c 即可
    "cd ../../../..","pwd",
    "^C",
])

lines = []
build_tree({"p":{"q":{"r":{"s":{"leaf":None}},"extra":None}}},lines)
lines += [
    "cd p/q/r/s","pwd",
    "touch a","touch b",
    "rm ./a","rm ../s/b","ls",
    "cd ../../..","pwd",       # /p
    "rm q/r/s/leaf",
    "ls",
    "cd ..","ls",              # /
    "rm p/q/extra",
    "touch .dot","la",
    "^C",
]
save(2,2,lines)

random.seed(7)
depth=7; chain=[rname(2,4) for _ in range(depth)]
lines=[]
for d in chain: lines += [f"mkdir {d}",f"cd {d}"]
lines.append("pwd")
# 每輪：先從當前位置往上走 up 層（相對路徑測試），
# 再從根往下走 down 層（用絕對路徑避免 cur 不確定）
cur_depth = depth
for _ in range(40):
    if cur_depth == 0:
        # 在根目錄，用相對路徑往下走
        down = random.randint(1, depth)
        lines.append("cd " + "/".join(chain[:down]))
        cur_depth = down
        lines.append("pwd")
        continue
    up = random.randint(1, cur_depth)
    lines.append("cd " + "/".join([".."]*up))
    cur_depth -= up
    lines.append("pwd")
    down = random.randint(0, depth - cur_depth)
    if down > 0:
        # 相對路徑：從 cur_depth 接著往下（偶爾混入 . 增加難度）
        segs = chain[cur_depth:cur_depth + down]
        if random.random() < 0.3:
            segs = ["."] + segs
        lines.append("cd " + "/".join(segs))
        cur_depth += down
    lines.append("pwd")
lines.append("^C")
save(2,3,lines)

# ── S3 絕對路徑 + rm -r ────────────────────────────────────────────────────────
print("S3")
lines=[]
build_tree({"x":{"a":{"b":{"f":None}}},"y":{"z":{"g":None}}},lines)
lines+=[
    "cd /x/a/b","pwd","ls",
    "cd /y/z","pwd","ls",
    "cd /x/a","pwd",
    "rm -r /y","cd /","ls",
    "cd /x/a/b","pwd",
    "rm -r /x/a",
    "pwd","ls",                # cur 退至 /x
    "cd /","ls",
    "rm -r /x","ls",
    "^C",
]
save(3,1,lines)

lines=[]
build_tree({"a":{"c":{"file":None}},"b":{}},lines)
lines+=[
    "cd /a/c","pwd",
    "rm -r /a/c","pwd","ls",   # cur 退至 /a
    "cd /a","ls",
    "rm -r /b","ls",
    "cd /","mkdir d","cd /d",
    "rm -r /d","pwd","ls",     # cur 退至 /
    "^C",
]
save(3,2,lines)

random.seed(99)
ROOTS=[rname(3,5) for _ in range(5)]
lines=[]
for r in ROOTS:
    build_tree({r:{rname(2,4):{rname(2,4):None,rname(2,4):None} for _ in range(3)}},lines)
for _ in range(15):
    r=random.choice(ROOTS)
    lines+=[f"cd /{r}","ls","pwd"]
for r in ROOTS[:2]: lines.append(f"rm -r /{r}")
lines+=["cd /","ls","^C"]
save(3,3,lines)

# ── S4 mv ─────────────────────────────────────────────────────────────────────
print("S4")
lines=[]
build_tree({"src":{"file1":None,"file2":None,"sub":{"deep":None}},"dst":{}},lines)
lines+=[
    "mv src/file1 src/renamed",
    "cd src","ls","cd /",
    "mv /src/renamed /dst/moved",
    "cd /dst","ls",
    "mv /src/sub /dst",
    "cd /dst","ls",
    "cd sub","ls","pwd",
    "mv /dst /top",
    "cd /top","ls","pwd",
    "cd /top/sub","pwd",
    "mv /src/file2 ./f2","ls",
    "cd /","ls",
    "^C",
]
save(4,1,lines)

lines=[]
build_tree({"box":{"coin":None},"trap":{},"gem":None},lines)
lines+=[
    "mv box trap",             # 搬入 trap/
    "cd trap","ls",
    "cd /trap/box","ls","pwd",
    "mv /trap/box /trap/renamed",
    "cd /trap","ls",
    "mv /trap/renamed /basket",
    "cd /","ls",
    "^C",
]
save(4,2,lines)

random.seed(123)
nodes=[f"n{i}" for i in range(8)]
lines=[]
for n in nodes: build_tree({n:{f"f{n}{j}":None for j in range(3)}},lines)
lines.append("ls")
cur_names=list(nodes)
for _ in range(8):
    src=random.choice(cur_names)
    new="r"+rname(2,3)
    while new in cur_names: new="r"+rname(2,3)
    lines.append(f"mv /{src} /{new}")
    cur_names[cur_names.index(src)]=new
lines.append("ls")
target=cur_names[0]
for n in cur_names[1:4]: lines.append(f"mv /{n} /{target}")
lines+=[f"cd /{target}","ls","pwd"]
for n in cur_names[4:]: lines.append(f"rm -r /{n}")
lines+=["cd /","ls","^C"]
save(4,3,lines)

print("\n完成！")