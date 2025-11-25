#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズムによるニューラルネットワーク学習 - 時計回り・ラップ適正化版
"""

import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, cpu_count
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# matplotlibをインタラクティブモードに
plt.ion()

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("⚡ Numba JIT compiler enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba not available, install with: pip install numba")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ==========================================
# パラメータ設定
# ==========================================
POP_SIZE = 200
GENERATIONS = 500
ELITE_SIZE = 15     # エリート保存数を少し増加
CXPB = 0.6          # 交叉確率
MUTPB_INITIAL = 0.4 # 初期変異確率（探索重視）
MUTPB_FINAL = 0.1   # 最終変異確率（収束重視）

NSENS = 7           # センサー数を5→7に増やして視野を拡大
NIN = NSENS + 1
NHID = 16           # 隠れ層を強化
NOUT = 2

SIM_STEPS = 2500    # シミュレーションステップ数を増加（周回しやすくする）
DT = 0.05
MAX_SENSOR_DIST = 50.0
# センサー角度（より広角に）
SENSOR_ANGLES = np.array([-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2])

WHEELBASE = 0.5
MAX_SPEED = 10.0    # 最高速度アップ
MAX_STEER = 0.6
THROTTLE_POWER = 4.0

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 50
REALTIME_UPDATE_INTERVAL = 5

# グローバル変数
TRACK = None
HALF_WIDTH = None
POINTS_PER_LAP = None # 1周あたりのポイント数
N_WEIGHTS = NHID * NIN + NHID + NOUT * NHID + NOUT

# ==========================================
# コース生成（時計回り対応）
# ==========================================
def generate_track(track_type="circuit", n_points_per_lap=200, total_laps=3):
    """
    時計回り(Clockwise)のコースを生成
    配列のインデックス順(0->1->2...)が進むべき方向となります。
    """
    # 時計回りにするために t の符号を反転、または sin 成分を反転させます
    # ここでは t をマイナス方向に進めることで時計回りを生成します
    t = np.linspace(0, -2 * np.pi * total_laps, n_points_per_lap * total_laps)
    
    if track_type == "circuit":
        r = 70.0 + 30.0 * np.sin(3.0 * t) + 15.0 * np.cos(7.0 * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        width = 14.0
    
    elif track_type == "oval":
        a, b = 100.0, 50.0
        x = a * np.cos(t) + 20.0 * np.sin(5.0 * t)
        y = b * np.sin(t) + 10.0 * np.cos(5.0 * t)
        width = 16.0
    
    elif track_type == "figure8":
        # 8の字は交差があるため難易度が高いが、tの進行方向で制御
        t_8 = np.linspace(0, -4 * np.pi * total_laps, n_points_per_lap * total_laps)
        r = 60.0
        x = r * np.sin(t_8)
        y = r * np.sin(t_8) * np.cos(t_8)
        width = 12.0
    
    else:
        return generate_track("circuit", n_points_per_lap, total_laps)
        
    return np.column_stack([x, y]), width, n_points_per_lap

# ==========================================
# Numba JIT関数群
# ==========================================
@jit(nopython=True, cache=True)
def point_to_segment_dist_jit(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c = vx * vx + vy * vy
    if c < 1e-10:
        return math.sqrt((px - ax)**2 + (py - ay)**2)
    t = (vx * wx + vy * wy) / c
    t = max(0.0, min(1.0, t))
    closest_x = ax + t * vx
    closest_y = ay + t * vy
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True, cache=True)
def distance_to_track_jit(x, y, track):
    min_dist = 1e9
    # 探索範囲を絞るための簡易的な最適化も可能だが、
    # ここでは安全性のため全探索（Numbaなら十分高速）
    for i in range(len(track) - 1):
        d = point_to_segment_dist_jit(x, y,
                                      track[i, 0], track[i, 1],
                                      track[i+1, 0], track[i+1, 1])
        if d < min_dist:
            min_dist = d
    return min_dist

@jit(nopython=True, cache=True)
def get_nearest_idx_jit(x, y, track, last_idx, look_ahead=50):
    """
    前回位置周辺を優先探索して高速化
    """
    n_points = len(track)
    best_idx = -1
    min_dist = 1e18

    # 初回または見失った場合
    start = 0
    end = n_points
    
    # 前回位置がわかっていれば、その周辺（前後）だけ探す
    if last_idx != -1:
        # 探索範囲設定（循環考慮）
        search_indices = np.arange(last_idx - 10, last_idx + look_ahead)
        
        for i in search_indices:
            # インデックスの正規化
            idx = i % n_points
            dx = x - track[idx, 0]
            dy = y - track[idx, 1]
            d = dx*dx + dy*dy
            if d < min_dist:
                min_dist = d
                best_idx = idx
    else:
        # 全探索
        for i in range(n_points):
            dx = x - track[i, 0]
            dy = y - track[i, 1]
            d = dx*dx + dy*dy
            if d < min_dist:
                min_dist = d
                best_idx = i
                
    return best_idx

@jit(nopython=True, cache=True)
def sense_jit(x, y, theta, track, half_width, angles, max_dist):
    readings = np.zeros(len(angles))
    for idx in range(len(angles)):
        angle = theta + angles[idx]
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        # レイキャストのステップサイズ
        step = 2.0
        d = 0.0
        hit = False
        
        # 簡易レイキャスト
        while d <= max_dist:
            px = x + dx * d
            py = y + dy * d
            dist_to_center = distance_to_track_jit(px, py, track)
            if dist_to_center > half_width:
                hit = True
                break
            d += step
            
        readings[idx] = (d if hit else max_dist) / max_dist
    return readings

@jit(nopython=True, cache=True)
def nn_forward_jit(weights, inputs, nin, nhid, nout):
    # Layer 1
    w1_end = nin * nhid
    w1 = weights[0:w1_end].reshape((nhid, nin))
    b1 = weights[w1_end:w1_end + nhid]
    
    # Layer 2
    w2_start = w1_end + nhid
    w2_end = w2_start + nhid * nout
    w2 = weights[w2_start:w2_end].reshape((nout, nhid))
    b2 = weights[w2_end:]
    
    # Forward
    # Hidden (ReLU)
    hidden = np.zeros(nhid)
    for i in range(nhid):
        val = 0.0
        for j in range(nin):
            val += w1[i, j] * inputs[j]
        val += b1[i]
        hidden[i] = max(0.0, val)
        
    # Output (Tanh)
    output = np.zeros(nout)
    for i in range(nout):
        val = 0.0
        for j in range(nhid):
            val += w2[i, j] * hidden[j]
        val += b2[i]
        output[i] = math.tanh(val)
        
    return output

@jit(nopython=True, cache=True)
def simulate_car_jit(weights, track, half_width, points_per_lap, sensor_angles,
                     sim_steps, dt, max_speed, max_steer, throttle_power, wheelbase):
    """
    シミュレーション実行
    戻り値: fitness, lap_count (周回数 float)
    """
    x, y = track[0, 0], track[0, 1]
    # 初期向き：track[0] -> track[1] (時計回り生成されているのでこれでOK)
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0

    n_points = len(track)
    last_idx = 0
    total_idx_progress = 0 # 累積インデックス進捗
    max_idx_progress = 0   # 到達した最大進捗
    
    steps_alive = 0
    total_speed = 0.0
    
    # 逆走判定用
    wrong_way_count = 0

    for step in range(sim_steps):
        # センサー
        sensors = sense_jit(x, y, theta, track, half_width, sensor_angles, MAX_SENSOR_DIST)
        
        # NN
        inputs = np.zeros(len(sensors) + 1)
        for i in range(len(sensors)):
            inputs[i] = sensors[i]
        inputs[len(sensors)] = v / max_speed
        
        outputs = nn_forward_jit(weights, inputs, len(inputs), NHID, NOUT)
        
        steer = outputs[0] * max_steer
        throttle = outputs[1] # -1 to 1

        # 物理
        if throttle > 0:
            v += throttle * throttle_power * dt
        else:
            v += throttle * throttle_power * 2.0 * dt # ブレーキは強く
            
        v = max(-2.0, min(max_speed, v)) # バックは遅く
        
        theta += (v * math.tan(steer) / wheelbase) * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        
        # コースアウト判定
        dist = distance_to_track_jit(x, y, track)
        if dist > half_width:
            break
            
        steps_alive += 1
        total_speed += v
        
        # 進捗計算（時計回り＝インデックス増加方向）
        current_idx = get_nearest_idx_jit(x, y, track, last_idx)
        
        # インデックスの差分計算（循環考慮）
        diff = current_idx - last_idx
        # 大きなジャンプ（ラップ境界）の補正
        if diff < -n_points / 2: # 例: 199 -> 2 (前進でラップまたぎ)
            diff += n_points
        elif diff > n_points / 2: # 例: 2 -> 199 (逆走でラップまたぎ)
            diff -= n_points
            
        if diff > 0:
            total_idx_progress += diff
            # 逆走カウンターリセット
            wrong_way_count = 0
        elif diff < 0:
            total_idx_progress += diff # 進捗を減らす（ペナルティ）
            wrong_way_count += 1
            
        if total_idx_progress > max_idx_progress:
            max_idx_progress = total_idx_progress
            
        last_idx = current_idx
        
        # 逆走し続けたら強制終了
        if wrong_way_count > 50:
            break

    # 評価関数の計算
    # 1. 距離スコア: 1周分のポイント数で正規化せず、純粋に進んだ距離（周回数）を評価
    laps_completed = max_idx_progress / points_per_lap
    
    # 2. 速度スコア: 生き残った時間に対する平均速度
    avg_speed = total_speed / steps_alive if steps_alive > 0 else 0
    
    # フィットネス計算
    # 周回数を最重要視（1周=100点換算的な重み付け）
    fitness = (laps_completed * 100.0) + (avg_speed * 2.0)
    
    # 早期死亡ペナルティ（ほとんど進まなかった場合）
    if steps_alive < 20:
        fitness = 0.0
        
    return fitness, laps_completed

# ==========================================
# リアルタイム可視化
# ==========================================
class RealtimeVisualizer:
    def __init__(self, track, half_width):
        self.track = track
        self.half_width = half_width
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 6), facecolor='#1e1e1e')
        
        # グラフ1: 進化ログ
        self.ax_fit = self.axes[0]
        self.ax_fit.set_facecolor('#2d2d2d')
        self.ax_fit.set_title("Fitness & Laps Evolution", color='white')
        self.ax_fit.set_xlabel("Generation", color='white')
        self.ax_fit.grid(True, alpha=0.3)
        self.ax_fit.tick_params(colors='white')
        
        # グラフ2: 軌跡
        self.ax_traj = self.axes[1]
        self.ax_traj.set_facecolor('#2d2d2d')
        self.ax_traj.set_title(f"Best Trajectory (Clockwise)", color='white')
        self.ax_traj.set_aspect('equal')
        self.ax_traj.tick_params(colors='white')
        
        # コース描画
        self.ax_traj.plot(track[:,0], track[:,1], c='#555555', lw=1, ls='--')
        # 内側と外側の境界線（可視化用）
        # 簡易的に円を描画
        step = 10
        for i in range(0, len(track), step):
            c = plt.Circle(track[i], half_width, color='#333333', alpha=0.3)
            self.ax_traj.add_artist(c)
            
        plt.tight_layout()
        plt.pause(0.1)

    def update(self, gen_log, best_weights, points_per_lap):
        # グラフ更新
        self.ax_fit.clear()
        self.ax_fit.set_facecolor('#2d2d2d')
        self.ax_fit.grid(True, alpha=0.3)
        
        gens = [x['gen'] for x in gen_log]
        fits = [x['max_fit'] for x in gen_log]
        laps = [x['max_laps'] for x in gen_log]
        
        # 2軸グラフ
        l1 = self.ax_fit.plot(gens, fits, color='#00ff00', label='Fitness')
        self.ax_fit.set_ylabel('Fitness', color='#00ff00')
        self.ax_fit.tick_params(axis='y', labelcolor='#00ff00')
        
        ax2 = self.ax_fit.twinx()
        l2 = ax2.plot(gens, laps, color='#00ffff', label='Laps')
        ax2.set_ylabel('Laps Completed', color='#00ffff')
        ax2.tick_params(axis='y', labelcolor='#00ffff')
        
        # 軌跡更新
        # 古い軌跡ラインを消すのは面倒なので、axesごとクリアしてトラック再描画
        self.ax_traj.clear()
        self.ax_traj.set_facecolor('#2d2d2d')
        self.ax_traj.set_title(f"Gen {gens[-1]}: {laps[-1]:.2f} Laps", color='white')
        self.ax_traj.plot(self.track[:,0], self.track[:,1], c='#555555', lw=1, ls='--')
        
        # ベスト個体のシミュレーション
        traj = simulate_trajectory(best_weights, self.track, self.half_width, points_per_lap)
        if len(traj) > 0:
            tx, ty = zip(*traj)
            self.ax_traj.plot(tx, ty, color='#ffcc00', lw=2, alpha=0.9)
            # スタート地点
            self.ax_traj.plot(tx[0], ty[0], 'o', c='lime', markersize=5)
            # 終了地点
            self.ax_traj.plot(tx[-1], ty[-1], 'x', c='red', markersize=8)
            
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

def simulate_trajectory(weights, track, half_width, points_per_lap):
    """可視化用の軌跡計算（JITなし）"""
    x, y = track[0, 0], track[0, 1]
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0
    trajectory = [(x, y)]
    
    for _ in range(SIM_STEPS):
        readings = sense_jit(x, y, theta, track, half_width, SENSOR_ANGLES, MAX_SENSOR_DIST)
        
        inputs = np.append(readings, v / MAX_SPEED)
        outputs = nn_forward_jit(weights, inputs, len(inputs), NHID, NOUT)
        
        steer = outputs[0] * MAX_STEER
        throttle = outputs[1]
        
        if throttle > 0:
            v += throttle * THROTTLE_POWER * DT
        else:
            v += throttle * THROTTLE_POWER * 2.0 * DT
        v = max(-2.0, min(MAX_SPEED, v))
        
        theta += (v * math.tan(steer) / WHEELBASE) * DT
        x += v * math.cos(theta) * DT
        y += v * math.sin(theta) * DT
        
        if distance_to_track_jit(x, y, track) > half_width:
            break
            
        trajectory.append((x, y))
        
    return trajectory

# ==========================================
# GA関連
# ==========================================
def eval_individual(individual):
    weights = np.array(individual, dtype=np.float64)
    fitness, laps = simulate_car_jit(
        weights, TRACK, HALF_WIDTH, POINTS_PER_LAP, SENSOR_ANGLES,
        SIM_STEPS, DT, MAX_SPEED, MAX_STEER, THROTTLE_POWER, WHEELBASE
    )
    return (fitness,)

def main():
    global TRACK, HALF_WIDTH, POINTS_PER_LAP
    
    print("🚀 Starting Clockwise Evolution...")
    
    # 1. コース生成（時計回り）
    # 1周あたりのポイント数を指定、合計周回数分の長さを確保
    n_points_lap = 200
    TRACK, HALF_WIDTH, POINTS_PER_LAP = generate_track("circuit", n_points_per_lap=n_points_lap, total_laps=3)
    print(f"Track generated: {len(TRACK)} points ({POINTS_PER_LAP} points/lap, Clockwise)")

    # 2. DEAP設定
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.normal, 0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=N_WEIGHTS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    pool = Pool(processes=cpu_count())
    toolbox.register("map", pool.map)
    
    # 3. 初期化
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(ELITE_SIZE)
    
    viz = RealtimeVisualizer(TRACK, HALF_WIDTH)
    gen_log = []
    
    print(f"\n{'Gen':<5} | {'Max Fit':<10} | {'Avg Fit':<10} | {'Best Laps':<10} | {'Status'}")
    print("-" * 60)

    # 4. 進化ループ
    for gen in range(GENERATIONS):
        # 評価
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        hof.update(pop)
        
        # 統計
        fits = [ind.fitness.values[0] for ind in pop]
        best_ind = hof[0]
        # ベスト個体のラップ数（可視化用に再計算）
        _, best_laps = simulate_car_jit(
            np.array(best_ind), TRACK, HALF_WIDTH, POINTS_PER_LAP, SENSOR_ANGLES,
            SIM_STEPS, DT, MAX_SPEED, MAX_STEER, THROTTLE_POWER, WHEELBASE
        )
        
        log_entry = {
            'gen': gen,
            'max_fit': max(fits),
            'avg_fit': sum(fits) / len(pop),
            'max_laps': best_laps
        }
        gen_log.append(log_entry)
        
        # ログ出力（ラップ数を明確に表示）
        print(f"{gen:<5} | {log_entry['max_fit']:<10.2f} | {log_entry['avg_fit']:<10.2f} | {best_laps:<10.2f} laps |", end="\r")
        
        # 可視化更新
        if gen % REALTIME_UPDATE_INTERVAL == 0:
            viz.update(gen_log, np.array(best_ind), POINTS_PER_LAP)
            
        # 次世代生成
        offspring = toolbox.select(pop, len(pop) - ELITE_SIZE)
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=CXPB, mutpb=MUTPB_FINAL)
        
        # エリート付加
        pop = list(hof) + list(offspring)
        
        # 目標達成したら終了（例: 2.5周したらOK）
        if best_laps >= 2.8:
            print(f"\n\n🎉 Target Reached! Completed {best_laps:.2f} laps.")
            break

    pool.close()
    pool.join()
    
    # 最終結果保存
    best_weights = np.array(hof[0])
    np.save("best_racer_clockwise.npy", best_weights)
    print("\n💾 Saved best weights.")
    
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()