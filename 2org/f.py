#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
タイムアタック型 AIレーシング (Spa風・複合コーナーコース版)
特徴:
- スプライン補間による複雑なサーキット形状
- 右コーナーと左コーナーの混在 (S字、シケイン)
- 高速区間とテクニカル区間のメリハリ
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings('ignore')
plt.ion()

try:
    from numba import jit
    NUMBA_AVAILABLE = True
    print("⚡ Numba JIT enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️ Numba not found.")
    def jit(*args, **kwargs):
        def decorator(func): return func
        return decorator

# ==========================================
# パラメータ設定
# ==========================================
POP_SIZE = 150
GENERATIONS = 300      # 難易度が上がったので世代数を増加
ELITE_SIZE = 12

NSENS = 7
NIN = NSENS + 1
NHID = 20              # 複雑なコースに対応するためニューロン数を増加
NOUT = 2
N_WEIGHTS = NHID * NIN + NHID + NOUT * NHID + NOUT

SIM_STEPS = 30000       # コースが長いので制限時間を延長
DT = 0.1
MAX_SENSOR_DIST = 60.0 # 視野を少し広く
SENSOR_ANGLES = np.array([-1.2, -0.7, -0.3, 0.0, 0.3, 0.7, 1.2])

WHEELBASE = 2.5  # m
MAX_SPEED = 300*1000/3600  # m/s
MAX_STEER = 0.6
THROTTLE_POWER = 10  # m/s**2

# グローバル変数
TRACK = None
HALF_WIDTH = None
POINTS_PER_LAP = None
GOAL_LINE = None

# ==========================================
# コース生成 (Catmull-Rom スプライン)
# ==========================================
def catmull_rom_spline(P0, P1, P2, P3, n_points=20):
    """2点間(P1-P2)を補間する曲線ポイントを生成"""
    t = np.linspace(0, 1, n_points)
    t2 = t * t
    t3 = t2 * t
    
    # Catmull-Rom 行列
    # 0.5 * [ -1  3 -3  1 ]
    #       [  2 -5  4 -1 ]
    #       [ -1  0  1  0 ]
    #       [  0  2  0  0 ]
    
    c0 = -0.5*t3 + t2 - 0.5*t
    c1 =  1.5*t3 - 2.5*t2 + 1.0
    c2 = -1.5*t3 + 2.0*t2 + 0.5*t
    c3 =  0.5*t3 - 0.5*t2
    
    x = c0*P0[0] + c1*P1[0] + c2*P2[0] + c3*P3[0]
    y = c0*P0[1] + c1*P1[1] + c2*P2[1] + c3*P3[1]
    
    return np.column_stack([x, y])

def generate_complex_track():
    """
    制御点を5倍に拡大し、実車が走れる広さと長さを確保
    """
    scale = 2.0  # スケール倍率 (これによりストレートが100m→500m級になる)
    
    # 形状データ (Spa風)
    base_waypoints = np.array([
        [0, -20], [-30, -20], [-20, 20], [-10, 30], [10, 40],
        [80, 80], [100, 70], [90, 50], [110, 30], [90, -10],
        [100, -50], [60, -70], [20, -60], [0, -40],
    ])
    
    # 座標拡大
    waypoints = base_waypoints * scale
    
    points = []
    n_waypoints = len(waypoints)
    for i in range(n_waypoints):
        p0 = waypoints[(i - 1) % n_waypoints]
        p1 = waypoints[i]
        p2 = waypoints[(i + 1) % n_waypoints]
        p3 = waypoints[(i + 2) % n_waypoints]
        # 点の密度も距離に応じて調整
        segment = catmull_rom_spline(p0, p1, p2, p3, n_points=50) 
        points.append(segment)
        
    track = np.vstack(points)
    
    # 道幅: 片側 7.5m (全幅 15m) -> F1開催サーキット基準
    width = 7.5 
    return track, width, len(track)



def calculate_track_poly(track, half_width):
    """壁の座標計算 (交差対策: ベクトル平滑化)"""
    # 勾配計算の前に少し平滑化すると壁がきれいになる
    pad = 5
    track_padded = np.vstack([track[-pad:], track, track[:pad]])
    
    # 移動平均で少しノイズ除去
    window = 3
    track_smooth = np.copy(track_padded)
    for i in range(window, len(track_padded)-window):
        track_smooth[i] = np.mean(track_padded[i-1:i+2], axis=0)
        
    dx = np.gradient(track_smooth[:, 0])
    dy = np.gradient(track_smooth[:, 1])
    
    normals = np.column_stack((-dy, dx))
    norm_lengths = np.linalg.norm(normals, axis=1)
    norm_lengths[norm_lengths == 0] = 1.0
    normals = normals / norm_lengths[:, np.newaxis]
    normals = normals[pad:-pad]
    
    # 壁生成
    inner = track + normals * half_width
    outer = track - normals * half_width
    poly_points = np.concatenate([outer, inner[::-1]])
    
    # ゴールライン
    goal_line = (inner[0,0], inner[0,1], outer[0,0], outer[0,1])
    
    return poly_points, inner, outer, goal_line

# ==========================================
# Numba JIT関数
# ==========================================
@jit(nopython=True, cache=True)
def is_intersect(p1_x, p1_y, p2_x, p2_y, q1_x, q1_y, q2_x, q2_y):
    def ccw(ax, ay, bx, by, cx, cy):
        return (by - ay) * (cx - ax) - (bx - ax) * (cy - ay)
    d1 = ccw(p1_x, p1_y, p2_x, p2_y, q1_x, q1_y)
    d2 = ccw(p1_x, p1_y, p2_x, p2_y, q2_x, q2_y)
    d3 = ccw(q1_x, q1_y, q2_x, q2_y, p1_x, p1_y)
    d4 = ccw(q1_x, q1_y, q2_x, q2_y, p2_x, p2_y)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

@jit(nopython=True, cache=True)
def distance_to_track_jit(x, y, track):
    min_dist = 1e9
    n = len(track)
    # 高速化: 探索ステップを飛ばす（詳細判定は近傍のみ）
    step = 2 
    for i in range(0, n, step):
        p1 = track[i]
        p2 = track[(i + 1) % n]
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        wx, wy = x - p1[0], y - p1[1]
        c = vx*vx + vy*vy
        if c < 1e-10:
            d = (x - p1[0])**2 + (y - p1[1])**2
        else:
            t = max(0.0, min(1.0, (vx*wx + vy*wy) / c))
            d = (x - (p1[0] + t*vx))**2 + (y - (p1[1] + t*vy))**2
        if d < min_dist: min_dist = d
    return math.sqrt(min_dist)

@jit(nopython=True, cache=True)
def get_nearest_idx_jit(x, y, track, last_idx):
    n = len(track)
    best_idx = -1
    min_dist = 1e18
    # 探索範囲を広げる（コースが複雑なため見失わないように）
    if last_idx == -1: search_range = range(0, n, 2)
    else: search_range = range(last_idx - 50, last_idx + 50)
    
    for i_raw in search_range:
        i = i_raw % n
        d = (x - track[i, 0])**2 + (y - track[i, 1])**2
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx

@jit(nopython=True, cache=True)
def sense_jit(x, y, theta, track, half_width, angles, max_dist):
    readings = np.zeros(len(angles))
    for i in range(len(angles)):
        angle = theta + angles[i]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        d = 0.0
        step = 5.0 # レイキャストの高速化
        hit = False
        while d < max_dist:
            px = x + d * cos_a
            py = y + d * sin_a
            if distance_to_track_jit(px, py, track) > half_width:
                hit = True
                break
            d += step
        # ヒットしたら二分探索で精度を上げる処理を簡易的に省略
        readings[i] = (d if hit else max_dist) / max_dist
    return readings

@jit(nopython=True, cache=True)
def nn_forward_jit(weights, inputs, nin, nhid, nout):
    idx = 0
    w1 = weights[idx : idx + nin*nhid].reshape((nhid, nin)); idx += nin*nhid
    b1 = weights[idx : idx + nhid]; idx += nhid
    w2 = weights[idx : idx + nhid*nout].reshape((nout, nhid)); idx += nhid*nout
    b2 = weights[idx : idx + nout]
    
    h = np.zeros(nhid)
    for i in range(nhid):
        val = b1[i]
        for j in range(nin): val += w1[i, j] * inputs[j]
        h[i] = max(0.0, val) # ReLU
    
    out = np.zeros(nout)
    for i in range(nout):
        val = b2[i]
        for j in range(nhid): val += w2[i, j] * h[j]
        out[i] = math.tanh(val) # Tanh
    return out

@jit(nopython=True, cache=True)
def simulate_car_jit(weights, track, half_width, points_per_lap, sensor_angles, 
                     sim_steps, dt, max_speed, max_steer, throttle_pwr, wheelbase,
                     goal_line_coords):
    x, y = track[0, 0], track[0, 1]
    # 初期向き: 1点目から2点目へのベクトル
    theta = math.atan2(track[3, 1] - y, track[3, 0] - x) # 少し先を見て安定化
    v = 0.0
    
    n_points = len(track)
    last_idx = 0
    total_idx = 0
    finished = False
    step = 0
    gx1, gy1, gx2, gy2 = goal_line_coords
    
    for s in range(sim_steps):
        step = s + 1
        prev_x, prev_y = x, y
        
        sensors = sense_jit(x, y, theta, track, half_width, sensor_angles, MAX_SENSOR_DIST)
        inputs = np.zeros(len(sensors) + 1)
        inputs[:len(sensors)] = sensors
        inputs[-1] = v / max_speed
        
        outputs = nn_forward_jit(weights, inputs, len(inputs), NHID, NOUT)
        steer = outputs[0] * max_steer
        throttle = outputs[1]
        
        if throttle > 0: v += throttle * throttle_pwr * dt
        else: v += throttle * throttle_pwr * 2.5 * dt # ブレーキ強化
        v = max(-3.0, min(max_speed, v))
        
        theta += (v * math.tan(steer) / wheelbase) * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        
        if distance_to_track_jit(x, y, track) > half_width: break
            
        curr = get_nearest_idx_jit(x, y, track, last_idx)
        diff = curr - last_idx
        if diff < -n_points / 2: diff += n_points
        elif diff > n_points / 2: diff -= n_points
        
        # 逆走ペナルティ (少し厳しく)
        if diff < 0: 
            total_idx += diff * 2
        else:
            total_idx += diff
            
        last_idx = curr
        
        # ゴール判定 (90%以上進んでゴールライン通過)
        if total_idx > points_per_lap * 0.9:
            if is_intersect(prev_x, prev_y, x, y, gx1, gy1, gx2, gy2):
                finished = True
                break

    if finished:
        fitness = 10000.0 + (sim_steps - step) * 5.0
    else:
        fitness = float(total_idx)
        if step < 50: fitness = 0.0

    return fitness, finished, float(step * dt)

# ==========================================
# 可視化クラス
# ==========================================
class Visualizer:
    def __init__(self, track, half_width, goal_line):
        self.track = track
        self.half_width = half_width
        self.goal_line = goal_line
        
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.patch.set_facecolor('#111111')
        
        self.ax_g = self.axes[0]
        self.ax_g.set_title('Learning Curve')
        self.ax_g.set_xlabel('Generation')
        self.ax_g.set_ylabel('Fitness')
        self.ax_g.grid(True, alpha=0.3)
        
        self.ax_c = self.axes[1]
        self.ax_c.set_aspect('equal')
        self.ax_c.axis('off')
        
        self.poly_points, self.border_in, self.border_out, _ = calculate_track_poly(track, half_width)
        self.road_poly = Polygon(self.poly_points, facecolor='#444444', edgecolor='none')
        self.ax_c.add_patch(self.road_poly)
        self.ax_c.plot(self.border_in[:,0], self.border_in[:,1], color='white', lw=1, alpha=0.8)
        self.ax_c.plot(self.border_out[:,0], self.border_out[:,1], color='white', lw=1, alpha=0.8)
        
        gx1, gy1, gx2, gy2 = goal_line
        self.ax_c.plot([gx1, gx2], [gy1, gy2], color='#ff0055', lw=3, zorder=10)
        self.ax_c.plot(track[0,0], track[0,1], 'o', color='lime', markersize=6, zorder=11)
        
        plt.tight_layout()

    def update(self, history, best_weights, points_per_lap):
        self.ax_g.clear()
        self.ax_g.grid(True, alpha=0.3)
        gens = [h['gen'] for h in history]
        fits = [h['fit'] for h in history]
        times = [h['time'] for h in history]
        
        self.ax_g.plot(gens, fits, color='cyan', label='Fitness')
        self.ax_g.legend(loc='upper left')
        ax2 = self.ax_g.twinx()
        ax2.plot(gens, times, color='yellow', linestyle='--', alpha=0.5, label='Time(s)')
        
        self.ax_c.clear()
        self.ax_c.set_aspect('equal')
        self.ax_c.axis('off')
        self.ax_c.add_patch(Polygon(self.poly_points, facecolor='#333333', edgecolor='none'))
        self.ax_c.plot(self.border_in[:,0], self.border_in[:,1], color='white', lw=1)
        self.ax_c.plot(self.border_out[:,0], self.border_out[:,1], color='white', lw=1)
        
        gx1, gy1, gx2, gy2 = self.goal_line
        self.ax_c.plot([gx1, gx2], [gy1, gy2], color='#ff0055', lw=3, zorder=10)
        self.ax_c.plot(self.track[0,0], self.track[0,1], 'o', color='lime', markersize=5, zorder=11)
        
        traj = self.run_sim(best_weights, points_per_lap)
        if len(traj) > 0:
            tx, ty = zip(*traj)
            self.ax_c.plot(tx, ty, color='orange', lw=2)
            
        last = history[-1]
        status = f"Gen {last['gen']} | Time: {last['time']:.2f}s" if last['finished'] else f"Gen {last['gen']} | Best Progress: {int(last['fit'])} pts"
        self.ax_c.set_title(status, color='white')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def run_sim(self, weights, points_per_lap):
        # 可視化用簡易シミュレーション
        x, y = self.track[0,0], self.track[0,1]
        theta = math.atan2(self.track[3,1]-y, self.track[3,0]-x)
        v = 0.0
        traj = []
        idx = 0
        total_p = 0
        gx1, gy1, gx2, gy2 = self.goal_line
        
        for _ in range(SIM_STEPS):
            prev_x, prev_y = x, y
            traj.append((x,y))
            sensors = sense_jit(x, y, theta, self.track, self.half_width, SENSOR_ANGLES, MAX_SENSOR_DIST)
            inp = np.zeros(len(sensors)+1)
            inp[:len(sensors)] = sensors
            inp[-1] = v / MAX_SPEED
            out = nn_forward_jit(weights, inp, len(inp), NHID, NOUT)
            steer = out[0] * MAX_STEER
            throttle = out[1]
            if throttle > 0: v += throttle * THROTTLE_POWER * DT
            else: v += throttle * THROTTLE_POWER * 2.5 * DT
            v = max(-3.0, min(MAX_SPEED, v))
            theta += (v * math.tan(steer) / WHEELBASE) * DT
            x += v * math.cos(theta) * DT
            y += v * math.sin(theta) * DT
            
            if distance_to_track_jit(x, y, self.track) > self.half_width: break
            
            cur = get_nearest_idx_jit(x, y, self.track, idx)
            diff = cur - idx
            n = len(self.track)
            if diff < -n/2: diff += n
            elif diff > n/2: diff -= n
            if diff > 0: total_p += diff
            idx = cur
            
            if total_p > points_per_lap * 0.9:
                if is_intersect(prev_x, prev_y, x, y, gx1, gy1, gx2, gy2):
                    traj.append((x,y))
                    break
        return traj

# ==========================================
# メイン
# ==========================================
def eval_wrapper(ind):
    w = np.array(ind)
    fit, _, _ = simulate_car_jit(
        w, TRACK, HALF_WIDTH, POINTS_PER_LAP, SENSOR_ANGLES, 
        SIM_STEPS, DT, MAX_SPEED, MAX_STEER, THROTTLE_POWER, WHEELBASE,
        GOAL_LINE
    )
    return fit,

def main():
    global TRACK, HALF_WIDTH, POINTS_PER_LAP, GOAL_LINE
    
    print("🚧 Generating Complex Track (Spline Interpolation)...")
    TRACK, HALF_WIDTH, POINTS_PER_LAP = generate_complex_track()
    _, _, _, GOAL_LINE = calculate_track_poly(TRACK, HALF_WIDTH)
    
    print(f"✅ Track Generated: {POINTS_PER_LAP} points. Width: {HALF_WIDTH}")
    print("   Includes S-curves, hairpins, and high-speed sections.")

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal, 0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N_WEIGHTS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.25, indpb=0.25)
    toolbox.register("select", tools.selTournament, tournsize=4)
    
    pool = Pool(cpu_count())
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(ELITE_SIZE)
    
    viz = Visualizer(TRACK, HALF_WIDTH, GOAL_LINE)
    history = []
    
    print("-" * 50)
    print(f"{'Gen':<5} | {'Best Fit':<10} | {'Time(s)':<10} | {'Status'}")
    print("-" * 50)
    
    for gen in range(1, GENERATIONS + 1):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.6, mutpb=0.15)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits): ind.fitness.values = fit
        pop = toolbox.select(offspring + list(hof), POP_SIZE)
        hof.update(pop)
        
        best_ind = hof[0]
        fit_val, finished, time_sec = simulate_car_jit(
            np.array(best_ind), TRACK, HALF_WIDTH, POINTS_PER_LAP, SENSOR_ANGLES, 
            SIM_STEPS, DT, MAX_SPEED, MAX_STEER, THROTTLE_POWER, WHEELBASE,
            GOAL_LINE
        )
        history.append({'gen': gen, 'fit': fit_val, 'time': time_sec, 'finished': finished})
        
        status = "🏁 FINISHED" if finished else f"Running ({int(fit_val/POINTS_PER_LAP*100)}%)"
        print(f"{gen:<5} | {fit_val:<10.1f} | {time_sec:<10.2f} | {status}")
        
        if gen % 5 == 0:
            viz.update(history, np.array(best_ind), POINTS_PER_LAP)
            
    pool.close()
    pool.join()
    print("\nDone. Close window to exit.")
    plt.show(block=True)

if __name__ == "__main__":
    main()