#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
タイムアタック型 AIレーシング (ゴールライン通過判定版)
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
GENERATIONS = 200
ELITE_SIZE = 10

NSENS = 7
NIN = NSENS + 1
NHID = 16
NOUT = 2
N_WEIGHTS = NHID * NIN + NHID + NOUT * NHID + NOUT

SIM_STEPS = 2000
DT = 0.05
MAX_SENSOR_DIST = 50.0
SENSOR_ANGLES = np.array([-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0])

WHEELBASE = 0.5
MAX_SPEED = 16.0
MAX_STEER = 0.6
THROTTLE_POWER = 8.0 # 加速性能アップ

# グローバル変数
TRACK = None
HALF_WIDTH = None
POINTS_PER_LAP = None
GOAL_LINE = None # (x1, y1, x2, y2)

# ==========================================
# コース生成
# ==========================================
def generate_track(n_points=250):
    t = np.linspace(0, -2 * np.pi, n_points, endpoint=False)
    # 緩やかなカーブ
    r = 90.0 + 30.0 * np.sin(2.0 * t) + 15.0 * np.cos(3.0 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    track = np.column_stack([x, y])
    width = 14.0 
    return track, width, n_points

def calculate_track_poly(track, half_width):
    """壁の座標計算"""
    pad = 5
    track_padded = np.vstack([track[-pad:], track, track[:pad]])
    dx = np.gradient(track_padded[:, 0])
    dy = np.gradient(track_padded[:, 1])
    normals = np.column_stack((-dy, dx))
    norm_lengths = np.linalg.norm(normals, axis=1)
    norm_lengths[norm_lengths == 0] = 1.0
    normals = normals / norm_lengths[:, np.newaxis]
    normals = normals[pad:-pad]
    
    inner = track + normals * half_width
    outer = track - normals * half_width
    poly_points = np.concatenate([outer, inner[::-1]])
    
    # ゴールライン座標 (インデックス0の内壁と外壁)
    goal_line = (inner[0,0], inner[0,1], outer[0,0], outer[0,1])
    
    return poly_points, inner, outer, goal_line

# ==========================================
# Numba JIT関数 (交差判定追加)
# ==========================================
@jit(nopython=True, cache=True)
def is_intersect(p1_x, p1_y, p2_x, p2_y, q1_x, q1_y, q2_x, q2_y):
    """線分(p1,p2)と線分(q1,q2)が交差しているか判定"""
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
def get_dist_sq(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

@jit(nopython=True, cache=True)
def distance_to_track_jit(x, y, track):
    min_dist = 1e9
    n = len(track)
    for i in range(n):
        p1 = track[i]
        p2 = track[(i + 1) % n]
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        wx, wy = x - p1[0], y - p1[1]
        c = vx*vx + vy*vy
        if c < 1e-10:
            d = (x - p1[0])**2 + (y - p1[1])**2
        else:
            t = (vx*wx + vy*wy) / c
            t = max(0.0, min(1.0, t))
            cx = p1[0] + t*vx
            cy = p1[1] + t*vy
            d = (x - cx)**2 + (y - cy)**2
        if d < min_dist: min_dist = d
    return math.sqrt(min_dist)

@jit(nopython=True, cache=True)
def get_nearest_idx_jit(x, y, track, last_idx):
    n = len(track)
    best_idx = -1
    min_dist = 1e18
    if last_idx == -1: search_range = range(n)
    else: search_range = range(last_idx - 30, last_idx + 30)
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
        step = 4.0 
        hit = False
        while d < max_dist:
            px = x + d * cos_a
            py = y + d * sin_a
            if distance_to_track_jit(px, py, track) > half_width:
                hit = True
                break
            d += step
        readings[i] = (d if hit else max_dist) / max_dist
    return readings

@jit(nopython=True, cache=True)
def nn_forward_jit(weights, inputs, nin, nhid, nout):
    w1 = weights[0 : nin*nhid].reshape((nhid, nin))
    b1 = weights[nin*nhid : nin*nhid + nhid]
    w2 = weights[nin*nhid + nhid : nin*nhid + nhid + nhid*nout].reshape((nout, nhid))
    b2 = weights[nin*nhid + nhid + nhid*nout :]
    h = np.zeros(nhid)
    for i in range(nhid):
        val = b1[i]
        for j in range(nin): val += w1[i, j] * inputs[j]
        h[i] = max(0.0, val)
    out = np.zeros(nout)
    for i in range(nout):
        val = b2[i]
        for j in range(nhid): val += w2[i, j] * h[j]
        out[i] = math.tanh(val)
    return out

@jit(nopython=True, cache=True)
def simulate_car_jit(weights, track, half_width, points_per_lap, sensor_angles, 
                     sim_steps, dt, max_speed, max_steer, throttle_pwr, wheelbase,
                     goal_line_coords):
    """
    シミュレーション (線分交差による厳密なゴール判定)
    goal_line_coords: (gx1, gy1, gx2, gy2)
    """
    x, y = track[0, 0], track[0, 1]
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0
    
    n_points = len(track)
    last_idx = 0
    total_idx = 0
    finished = False
    step = 0
    
    gx1, gy1, gx2, gy2 = goal_line_coords
    
    for s in range(sim_steps):
        step = s + 1
        prev_x, prev_y = x, y # 移動前の位置を保存
        
        # 1. センサー & NN
        sensors = sense_jit(x, y, theta, track, half_width, sensor_angles, MAX_SENSOR_DIST)
        inputs = np.zeros(len(sensors) + 1)
        inputs[:len(sensors)] = sensors
        inputs[-1] = v / max_speed
        
        outputs = nn_forward_jit(weights, inputs, len(inputs), NHID, NOUT)
        steer = outputs[0] * max_steer
        throttle = outputs[1]
        
        # 2. 物理
        if throttle > 0: v += throttle * throttle_pwr * dt
        else: v += throttle * throttle_pwr * 2.0 * dt
        v = max(-3.0, min(max_speed, v))
        theta += (v * math.tan(steer) / wheelbase) * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        
        # 3. 衝突判定
        if distance_to_track_jit(x, y, track) > half_width: break
            
        # 4. 進捗更新（逆走チェック用）
        curr = get_nearest_idx_jit(x, y, track, last_idx)
        diff = curr - last_idx
        if diff < -n_points / 2: diff += n_points
        elif diff > n_points / 2: diff -= n_points
        if diff > 0: total_idx += diff
        last_idx = curr
        
        # 5. ゴールライン通過判定 (厳密)
        # 条件:
        # A. コースの90%以上を進んでいること (フライング防止)
        # B. 車の移動線分 (prev -> curr) が ゴール線分 (goal_in -> goal_out) と交差している
        if total_idx > points_per_lap * 0.9:
            if is_intersect(prev_x, prev_y, x, y, gx1, gy1, gx2, gy2):
                finished = True
                break

    # 評価
    if finished:
        # 早くゴールするほど高得点
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
        self.goal_line = goal_line # (x1, y1, x2, y2)
        
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.patch.set_facecolor('#111111')
        
        self.ax_g = self.axes[0]
        self.ax_g.set_title('Learning Curve')
        self.ax_g.set_xlabel('Generation')
        self.ax_g.set_ylabel('Fitness')
        self.ax_g.grid(True, alpha=0.3)
        
        self.ax_c = self.axes[1]
        self.ax_c.set_aspect('equal')
        self.ax_c.axis('off')
        
        # トラック描画
        self.poly_points, self.border_in, self.border_out, _ = calculate_track_poly(track, half_width)
        self.road_poly = Polygon(self.poly_points, facecolor='#444444', edgecolor='none')
        self.ax_c.add_patch(self.road_poly)
        self.ax_c.plot(self.border_in[:,0], self.border_in[:,1], color='white', lw=1, alpha=0.8)
        self.ax_c.plot(self.border_out[:,0], self.border_out[:,1], color='white', lw=1, alpha=0.8)
        
        # ゴールライン描画 (赤色)
        gx1, gy1, gx2, gy2 = goal_line
        self.ax_c.plot([gx1, gx2], [gy1, gy2], color='red', lw=3, label='Goal Line', zorder=10)
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
        
        # ゴールライン再描画
        gx1, gy1, gx2, gy2 = self.goal_line
        self.ax_c.plot([gx1, gx2], [gy1, gy2], color='#ff0055', lw=3, alpha=0.9, zorder=10)
        self.ax_c.plot(self.track[0,0], self.track[0,1], 'o', color='lime', markersize=5, zorder=11)
        
        traj = self.run_sim(best_weights, points_per_lap)
        if len(traj) > 0:
            tx, ty = zip(*traj)
            self.ax_c.plot(tx, ty, color='orange', lw=2)
            
        last = history[-1]
        status = f"Gen {last['gen']} | Time: {last['time']:.2f}s" if last['finished'] else f"Gen {last['gen']} | Crash"
        self.ax_c.set_title(status, color='white')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def run_sim(self, weights, points_per_lap):
        x, y = self.track[0,0], self.track[0,1]
        theta = math.atan2(self.track[1,1]-y, self.track[1,0]-x)
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
            else: v += throttle * THROTTLE_POWER * 2.0 * DT
            v = max(-3.0, min(MAX_SPEED, v))
            theta += (v * math.tan(steer) / WHEELBASE) * DT
            x += v * math.cos(theta) * DT
            y += v * math.sin(theta) * DT
            
            if distance_to_track_jit(x, y, self.track) > self.half_width: break
            
            # 進捗チェック
            cur = get_nearest_idx_jit(x, y, self.track, idx)
            diff = cur - idx
            n = len(self.track)
            if diff < -n/2: diff += n
            elif diff > n/2: diff -= n
            if diff > 0: total_p += diff
            idx = cur
            
            # 簡易ゴールチェック (可視化用)
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
        GOAL_LINE # ゴールライン座標を渡す
    )
    return fit,

def main():
    global TRACK, HALF_WIDTH, POINTS_PER_LAP, GOAL_LINE
    TRACK, HALF_WIDTH, POINTS_PER_LAP = generate_track(n_points=250)
    
    # ゴールライン座標の計算
    _, _, _, GOAL_LINE = calculate_track_poly(TRACK, HALF_WIDTH)
    
    print(f"Track created: {POINTS_PER_LAP} points. Clockwise.")
    print(f"Goal Line Set: {GOAL_LINE}")

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal, 0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N_WEIGHTS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pool = Pool(cpu_count())
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(ELITE_SIZE)
    
    # 可視化時にゴールラインも渡す
    viz = Visualizer(TRACK, HALF_WIDTH, GOAL_LINE)
    history = []
    
    print("-" * 50)
    print(f"{'Gen':<5} | {'Best Fit':<10} | {'Time(s)':<10} | {'Status'}")
    print("-" * 50)
    
    for gen in range(1, GENERATIONS + 1):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.6, mutpb=0.1)
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
        
        status = "🏁 FINISHED" if finished else "Running..."
        print(f"{gen:<5} | {fit_val:<10.1f} | {time_sec:<10.2f} | {status}")
        
        if gen % 5 == 0:
            viz.update(history, np.array(best_ind), POINTS_PER_LAP)
            
    pool.close()
    pool.join()
    print("\nDone. Close window to exit.")
    plt.show(block=True)

if __name__ == "__main__":
    main()