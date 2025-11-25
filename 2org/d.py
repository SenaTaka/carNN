#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
タイムアタック型 AIレーシング (1周終了・タイム評価版)
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

# NN構造
NSENS = 7
NIN = NSENS + 1
NHID = 16
NOUT = 2
N_WEIGHTS = NHID * NIN + NHID + NOUT * NHID + NOUT

# シミュレーション設定
SIM_STEPS = 2000       # 制限時間（これ以内に1周できないとリタイア）
DT = 0.05
MAX_SENSOR_DIST = 50.0
SENSOR_ANGLES = np.array([-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0])

# 車両物理
WHEELBASE = 0.5
MAX_SPEED = 15.0       # タイムアタック用に少し高速化
MAX_STEER = 0.6
THROTTLE_POWER = 6.0

# グローバル変数
TRACK = None
HALF_WIDTH = None
POINTS_PER_LAP = None

# ==========================================
# コース生成 & 境界線計算（修正版）
# ==========================================
def generate_track(n_points=250):
    """時計回りの単一周回コースを生成"""
    # 1周分だけ生成 (0 から -2pi)
    t = np.linspace(0, -2 * np.pi, n_points, endpoint=False) # endpoint=Falseで始点と終点の重複を防ぐ
    
    # コース形状 (Circuit)
    r = 80.0 + 30.0 * np.sin(3.0 * t) + 15.0 * np.cos(7.0 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    
    # 配列化
    track = np.column_stack([x, y])
    width = 16.0
    return track, width, n_points

def calculate_track_poly(track, half_width):
    """
    コースを「面」として描画するためのポリゴン座標を計算
    始点と終点を滑らかにつなぐ処理を追加
    """
    # 1. 勾配計算用にデータを循環させる (パディング)
    # これにより始点と終点の法線ベクトルが滑らかにつながる
    pad = 5
    track_padded = np.vstack([track[-pad:], track, track[:pad]])
    
    dx = np.gradient(track_padded[:, 0])
    dy = np.gradient(track_padded[:, 1])
    
    # 法線ベクトル
    normals = np.column_stack((-dy, dx))
    norm_lengths = np.linalg.norm(normals, axis=1)
    norm_lengths[norm_lengths == 0] = 1.0
    normals = normals / norm_lengths[:, np.newaxis]
    
    # パディング除去
    normals = normals[pad:-pad]
    
    # 境界線計算
    inner = track + normals * half_width
    outer = track - normals * half_width
    
    # ポリゴン用にデータを結合 (外側 -> 内側(逆順) -> 始点に戻る)
    # これで「穴の空いたドーナツ型」ではなく「一筆書きの閉じた図形」を作る
    poly_points = np.concatenate([outer, inner[::-1]])
    
    return poly_points, inner, outer

# ==========================================
# Numba JIT関数 (ロジック変更)
# ==========================================
@jit(nopython=True, cache=True)
def get_dist_sq(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

@jit(nopython=True, cache=True)
def distance_to_track_jit(x, y, track):
    min_dist = 1e9
    # 閉じたループとして扱うため、最後の点と最初の点の線分も考慮
    n = len(track)
    for i in range(n):
        p1 = track[i]
        p2 = track[(i + 1) % n] # 循環
        
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
            
        if d < min_dist:
            min_dist = d
    return math.sqrt(min_dist)

@jit(nopython=True, cache=True)
def get_nearest_idx_jit(x, y, track, last_idx):
    n = len(track)
    best_idx = -1
    min_dist = 1e18
    
    # 前回の位置周辺を探索
    if last_idx == -1:
        search_range = range(n)
    else:
        # 前後30ポイントを探索（循環考慮）
        search_range = range(last_idx - 30, last_idx + 30)
        
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
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
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
    # 重み展開と推論 (ReLU + Tanh)
    w1 = weights[0 : nin*nhid].reshape((nhid, nin))
    b1 = weights[nin*nhid : nin*nhid + nhid]
    w2 = weights[nin*nhid + nhid : nin*nhid + nhid + nhid*nout].reshape((nout, nhid))
    b2 = weights[nin*nhid + nhid + nhid*nout :]
    
    h = np.zeros(nhid)
    for i in range(nhid):
        val = b1[i]
        for j in range(nin):
            val += w1[i, j] * inputs[j]
        h[i] = max(0.0, val)
        
    out = np.zeros(nout)
    for i in range(nout):
        val = b2[i]
        for j in range(nhid):
            val += w2[i, j] * h[j]
        out[i] = math.tanh(val)
    return out

@jit(nopython=True, cache=True)
def simulate_car_jit(weights, track, half_width, points_per_lap, sensor_angles, 
                     sim_steps, dt, max_speed, max_steer, throttle_pwr, wheelbase):
    """
    シミュレーション (1周終了・タイム評価)
    """
    x, y = track[0, 0], track[0, 1]
    # 初期向き
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0
    
    n_points = len(track)
    last_idx = 0
    total_idx = 0
    
    finished = False
    crash = False
    step = 0
    
    # シミュレーションループ
    for s in range(sim_steps):
        step = s + 1
        
        # 1. センサー & NN
        sensors = sense_jit(x, y, theta, track, half_width, sensor_angles, MAX_SENSOR_DIST)
        inputs = np.zeros(len(sensors) + 1)
        inputs[:len(sensors)] = sensors
        inputs[-1] = v / max_speed
        
        outputs = nn_forward_jit(weights, inputs, len(inputs), NHID, NOUT)
        
        # 2. 物理
        steer = outputs[0] * max_steer
        throttle = outputs[1]
        
        if throttle > 0: v += throttle * throttle_pwr * dt
        else: v += throttle * throttle_pwr * 2.0 * dt
        v = max(-3.0, min(max_speed, v))
        
        theta += (v * math.tan(steer) / wheelbase) * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        
        # 3. 衝突判定
        if distance_to_track_jit(x, y, track) > half_width:
            crash = True
            break
            
        # 4. 進捗更新
        curr = get_nearest_idx_jit(x, y, track, last_idx)
        diff = curr - last_idx
        
        # ラップまたぎ補正
        if diff < -n_points / 2: diff += n_points
        elif diff > n_points / 2: diff -= n_points
        
        if diff > 0: total_idx += diff
        
        last_idx = curr
        
        # 5. ゴール判定 (1周 = points_per_lap)
        # 念のため少し余裕を持たせる(>=)
        if total_idx >= points_per_lap:
            finished = True
            break

    # ==============================
    # 評価関数 (ここを変更)
    # ==============================
    
    # ベースの距離点
    distance_score = total_idx
    
    if finished:
        # 完走ボーナス (5000点)
        # タイムボーナス (残りステップ数が多いほど高得点)
        # これにより「速いタイム」が「長い距離」より圧倒的に偉くなる
        time_bonus = (sim_steps - step) * 2.0
        fitness = 5000.0 + time_bonus
    else:
        # 完走していない場合、進んだ距離がスコア
        # ただし完走者の最低点(5000)を超えないようにする
        fitness = float(distance_score)
        # 早期クラッシュへのペナルティ
        if step < 50:
            fitness = 0.0

    return fitness, finished, float(step * dt)

# ==========================================
# 可視化クラス
# ==========================================
class Visualizer:
    def __init__(self, track, half_width):
        self.track = track
        self.half_width = half_width
        
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 6))
        self.fig.patch.set_facecolor('#111111')
        
        # 左：適応度グラフ
        self.ax_g = self.axes[0]
        self.ax_g.set_title('Learning Curve')
        self.ax_g.set_xlabel('Generation')
        self.ax_g.set_ylabel('Fitness (Time Score)')
        self.ax_g.grid(True, alpha=0.3)
        
        # 右：コース描画
        self.ax_c = self.axes[1]
        self.ax_c.set_aspect('equal')
        self.ax_c.axis('off')
        
        # --- コースの描画（ポリゴンで綺麗に）---
        # 計算
        self.poly_points, self.border_in, self.border_out = calculate_track_poly(track, half_width)
        
        # アスファルト部分（塗りつぶし）
        self.road_poly = Polygon(self.poly_points, facecolor='#444444', edgecolor='none')
        self.ax_c.add_patch(self.road_poly)
        
        # 白線（境界）
        self.ax_c.plot(self.border_in[:,0], self.border_in[:,1], color='white', lw=1, alpha=0.8)
        self.ax_c.plot(self.border_out[:,0], self.border_out[:,1], color='white', lw=1, alpha=0.8)
        
        # スタート地点
        self.ax_c.plot(track[0,0], track[0,1], 'o', color='lime', markersize=6)
        
        plt.tight_layout()

    def update(self, history, best_weights, points_per_lap):
        # グラフ更新
        self.ax_g.clear()
        self.ax_g.grid(True, alpha=0.3)
        gens = [h['gen'] for h in history]
        fits = [h['fit'] for h in history]
        times = [h['time'] for h in history]
        
        # Fitness
        self.ax_g.plot(gens, fits, color='cyan', label='Fitness')
        self.ax_g.legend(loc='upper left')
        
        # Time (完走タイム) - 右軸
        ax2 = self.ax_g.twinx()
        ax2.plot(gens, times, color='yellow', linestyle='--', alpha=0.5, label='Time(s)')
        ax2.set_ylabel('Best Lap Time (s)', color='yellow')
        
        # 軌跡描画
        # 背景（コース）は消さないようにLineだけ更新したいが、簡単のため再描画
        self.ax_c.clear()
        self.ax_c.set_aspect('equal')
        self.ax_c.axis('off')
        
        # コース再追加
        self.ax_c.add_patch(Polygon(self.poly_points, facecolor='#333333', edgecolor='none'))
        self.ax_c.plot(self.border_in[:,0], self.border_in[:,1], color='white', lw=1)
        self.ax_c.plot(self.border_out[:,0], self.border_out[:,1], color='white', lw=1)
        self.ax_c.plot(self.track[0,0], self.track[0,1], 'o', color='lime', markersize=5)
        
        # 走行シミュレーション（可視化用）
        traj = self.run_sim(best_weights, points_per_lap)
        if len(traj) > 0:
            tx, ty = zip(*traj)
            self.ax_c.plot(tx, ty, color='orange', lw=2)
            
        last = history[-1]
        status = f"Gen {last['gen']} | Time: {last['time']:.2f}s" if last['finished'] else f"Gen {last['gen']} | Crash (Dist: {int(last['fit'])})"
        self.ax_c.set_title(status, color='white')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def run_sim(self, weights, points_per_lap):
        # 可視化用の簡易シミュレータ（Python実装）
        x, y = self.track[0,0], self.track[0,1]
        theta = math.atan2(self.track[1,1]-y, self.track[1,0]-x)
        v = 0.0
        traj = []
        
        idx = 0
        total_p = 0
        
        for _ in range(SIM_STEPS):
            traj.append((x,y))
            
            # センサー等
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
            
            if distance_to_track_jit(x, y, self.track) > self.half_width:
                break
                
            # ゴール判定用簡易ロジック
            cur = get_nearest_idx_jit(x, y, self.track, idx)
            diff = cur - idx
            n = len(self.track)
            if diff < -n/2: diff += n
            elif diff > n/2: diff -= n
            if diff > 0: total_p += diff
            idx = cur
            
            if total_p >= points_per_lap:
                traj.append((x,y))
                break
                
        return traj

# ==========================================
# メイン
# ==========================================
def eval_wrapper(ind):
    w = np.array(ind)
    fit, _, _ = simulate_car_jit(w, TRACK, HALF_WIDTH, POINTS_PER_LAP, SENSOR_ANGLES, SIM_STEPS, DT, MAX_SPEED, MAX_STEER, THROTTLE_POWER, WHEELBASE)
    return fit,

def main():
    global TRACK, HALF_WIDTH, POINTS_PER_LAP
    
    # 1. コース作成
    TRACK, HALF_WIDTH, POINTS_PER_LAP = generate_track(n_points=250)
    print(f"Track created: {POINTS_PER_LAP} points. Clockwise.")

    # 2. GA準備
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
    
    viz = Visualizer(TRACK, HALF_WIDTH)
    history = []
    
    print("-" * 50)
    print(f"{'Gen':<5} | {'Best Fit':<10} | {'Time(s)':<10} | {'Status'}")
    print("-" * 50)
    
    # 3. ループ
    for gen in range(1, GENERATIONS + 1):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.6, mutpb=0.1)
        
        fits = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
            
        pop = toolbox.select(offspring + list(hof), POP_SIZE)
        hof.update(pop)
        
        # ベスト個体の詳細データ取得
        best_ind = hof[0]
        fit_val, finished, time_sec = simulate_car_jit(np.array(best_ind), TRACK, HALF_WIDTH, POINTS_PER_LAP, SENSOR_ANGLES, SIM_STEPS, DT, MAX_SPEED, MAX_STEER, THROTTLE_POWER, WHEELBASE)
        
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