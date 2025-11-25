# run_and_show.py
# Cコードの上書きはせず、コンパイル・実行・可視化のみを行う

import subprocess, struct, os, math
import matplotlib.pyplot as plt

TRACK_FN = "track.txt"
WEIGHTS_FN = "all_pop.weights"
EXE_NAME = "nn_evolve"
if os.name == 'nt': EXE_NAME += ".exe"

# ==========================================
# 1. コース生成
# ==========================================
def write_light_track(fn):
    pts = []
    n_points = 200 # 軽量化のため点数を抑える
    laps = 2
    for i in range(n_points * laps):
        t = (i / n_points) * 2 * math.pi
        # 複雑な形状 (R=70ベース)
        r = 70.0 + 30.0 * math.sin(3.0 * t) + 15.0 * math.cos(7.0 * t)
        x = r * math.cos(t)
        y = r * math.sin(t)
        pts.append((x, y))
    with open(fn,"w") as f:
        f.write(f"{len(pts)} 12.0\n") # 幅12.0
        for x,y in pts:
            f.write(f"{x:.6f} {y:.6f}\n")
    print(f"Generated track: {fn}")

# ==========================================
# 2. Cコードのコンパイルと実行
# ==========================================
def compile_c_code():
    if not os.path.exists("nn_evolve.c"):
        print("Error: nn_evolve.c not found.")
        exit(1)
    
    print("Compiling C code...")
    cmd = ["gcc", "-O3", "-lm", "-o", "nn_evolve", "nn_evolve.c"]
    try:
        subprocess.check_call(cmd)
        print("Compilation success.")
    except Exception as e:
        print(f"Compilation failed: {e}")
        exit(1)

def run_c_simulation():
    print("Running GA simulation (C code)...")
    try:
        cmd = [f"./{EXE_NAME}", TRACK_FN]
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Execution failed: {e}")
        exit(1)

# ==========================================
# 3. Python側の物理シミュレーション (Cと合わせる)
# ==========================================
def read_all_weights(fn, n_weights_per_genome):
    if not os.path.exists(fn): return []
    with open(fn,"rb") as f: data = f.read()
    total_doubles = len(data) // 8
    n_pop = total_doubles // n_weights_per_genome
    pop_weights = []
    for i in range(n_pop):
        start = i * n_weights_per_genome * 8
        end = start + n_weights_per_genome * 8
        chunk = data[start:end]
        pop_weights.append(struct.unpack(f"{n_weights_per_genome}d", chunk))
    return pop_weights

def nn_forward_py(weights, nin, nhid, nout, inputv):
    p=0
    hid = []
    for j in range(nhid):
        s=0.0
        for i in range(nin): s += weights[p]*inputv[i]; p+=1
        s += weights[p]; p+=1
        hid.append(math.tanh(s))
    out = []
    for k in range(nout):
        s=0.0
        for j in range(nhid): s += weights[p]*hid[j]; p+=1
        s += weights[p]; p+=1
        out.append(math.tanh(s))
    return out

def point_to_seg_dist(px,py,ax,ay,bx,by):
    vx, vy = bx-ax, by-ay
    wx, wy = px-ax, py-ay
    c = vx*vx+vy*vy
    if c==0: return math.hypot(px-ax,py-ay)
    t = max(0.0, min(1.0, (vx*wx+vy*wy)/c))
    return math.hypot(px-(ax+t*vx), py-(ay+t*vy))

def sense_py(track, carx, cary, carth, angles, maxdist, half_width):
    out=[]
    for a in angles:
        ang = carth + a
        dx, dy = math.cos(ang), math.sin(ang)
        d=0.0; step=2.0; hit=False 
        while d<=maxdist:
            px, py = carx + dx*d, cary + dy*d
            mind = 1e9
            for i in range(len(track)-1):
                pd = point_to_seg_dist(px,py, track[i][0],track[i][1],track[i+1][0],track[i+1][1])
                if pd<mind: mind=pd
            if mind > half_width: hit=True; break
            d += step
        out.append((d if hit else maxdist)/maxdist)
    return out

def simulate_one_car(track, half, weights, nsens, angles, nhid, nout, maxdist):
    nin = nsens + 1
    sx,sy = track[0]
    stheta = math.atan2(track[1][1]-sy, track[1][0]-sx)
    x,y,theta,v = sx,sy,stheta,0.0
    traj = []
    
    SIM_STEPS = 1500 
    for t in range(SIM_STEPS):
        sens = sense_py(track,x,y,theta,angles,maxdist,half)
        inp = sens + [v/8.0]
        out = nn_forward_py(weights, nin, nhid, nout, inp)
        steer = max(-1.0,min(1.0,out[0])) * 0.7
        throttle = out[1]
        
        v += max(-1.0,min(1.0,throttle)) * 3.5 * 0.05
        v = max(-1.0, min(8.0, v))
        theta += (v * math.tan(steer)/0.5) * 0.05
        x += v*math.cos(theta)*0.05; y += v*math.sin(theta)*0.05
        traj.append((x,y))
        
        mind=1e9
        for i in range(len(track)-1):
            pd = point_to_seg_dist(x,y, track[i][0],track[i][1],track[i+1][0],track[i+1][1])
            if pd<mind: mind=pd
        if mind > half: break
    return traj

def main():
    # 1. コース生成
    write_light_track(TRACK_FN)

    # 2. コンパイル & 実行
    compile_c_code()
    run_c_simulation()
    
    # 3. データの準備
    with open(TRACK_FN,"r") as f:
        header = f.readline().strip().split()
        n = int(header[0]); half = float(header[1])
        track=[]
        for _ in range(n):
            a=f.readline().split(); track.append((float(a[0]),float(a[1])))
            
    # ※注意: Cコードの設定と合わせる必要があります
    NSENS = 5
    ANGLES = [-1.0, -0.5, 0.0, 0.5, 1.0]
    NHID = 8
    MAXDIST = 40.0
    nin = NSENS+1; nout=2
    nw = NHID*nin + NHID + nout*NHID + nout
    
    all_weights = read_all_weights(WEIGHTS_FN, nw)
    if not all_weights: return

    print(f"Simulating {len(all_weights)} genomes (Sampling enabled)...")
    all_trajs = []
    best_traj = []
    
    count = 0
    for i, w in enumerate(all_weights):
        # 間引き描画 (ベストは必ず描く。他は3体に1体)
        if i != 0 and i % 3 != 0: 
            continue
            
        if count % 5 == 0: print(f"Simulating car index {i}...")
        traj = simulate_one_car(track, half, w, NSENS, ANGLES, NHID, nout, MAXDIST)
        
        if i == 0: best_traj = traj
        else: all_trajs.append(traj)
        count += 1

    # 4. 描画
    print("Plotting...")
    tx = [p[0] for p in track]; ty=[p[1] for p in track]
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    ax.plot(tx, ty, color='white', linewidth=1, alpha=0.3)
    for i in range(0, len(track), 3):
        c = plt.Circle(track[i], half, color='white', alpha=0.05)
        ax.add_artist(c)

    # 群衆 (シアン)
    for traj in all_trajs:
        px = [p[0] for p in traj]; py=[p[1] for p in traj]
        ax.plot(px, py, color='cyan', linewidth=1, alpha=0.15)
            
    # ベスト (赤)
    if best_traj:
        bpx = [p[0] for p in best_traj]; bpy=[p[1] for p in best_traj]
        ax.plot(bpx, bpy, color='red', linewidth=2.5, label='Best AI')

    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend()
    plt.title("Evolution Result (Light Mode)", color='white')
    plt.show()

if __name__=='__main__':
    main()