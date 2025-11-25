#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズムによるニューラルネットワーク学習 - 超高速版
DEAP + Numba + マルチプロセスで最大限高速化
"""

import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("⚡ Numba JIT compiler enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba not available, install with: pip install numba")
    # ダミーデコレータ
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
ELITE_SIZE = 20
CXPB = 0.7  # 交叉確率
MUTPB = 0.2  # 変異確率

NSENS = 5
NIN = NSENS + 1
NHID = 8
NOUT = 2

SIM_STEPS = 2000
DT = 0.05
MAX_SENSOR_DIST = 40.0
SENSOR_ANGLES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

WHEELBASE = 0.5
MAX_SPEED = 8.0
MAX_STEER = 0.7
THROTTLE_POWER = 3.5

# グローバル変数（マルチプロセスで共有）
TRACK = None
HALF_WIDTH = None
N_WEIGHTS = NHID * NIN + NHID + NOUT * NHID + NOUT

# ==========================================
# コース生成
# ==========================================
def generate_track(n_points=200, laps=2):
    """サーキットコース生成"""
    t = np.linspace(0, 2 * np.pi * laps, n_points * laps)
    r = 70.0 + 30.0 * np.sin(3.0 * t) + 15.0 * np.cos(7.0 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y]), 12.0

# ==========================================
# Numba JITコンパイル済み関数
# ==========================================
@jit(nopython=True, cache=True)
def point_to_segment_dist_jit(px, py, ax, ay, bx, by):
    """点から線分への距離（JIT最適化）"""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c = vx * vx + vy * vy
    if c < 1e-10:
        return math.sqrt((px - ax)**2 + (py - ay)**2)
    t = max(0.0, min(1.0, (vx * wx + vy * wy) / c))
    closest_x = ax + t * vx
    closest_y = ay + t * vy
    return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)

@jit(nopython=True, cache=True)
def distance_to_track_jit(x, y, track):
    """トラックまでの最短距離（JIT最適化）"""
    min_dist = 1e9
    for i in range(len(track) - 1):
        d = point_to_segment_dist_jit(x, y, 
                                       track[i, 0], track[i, 1],
                                       track[i+1, 0], track[i+1, 1])
        if d < min_dist:
            min_dist = d
    return min_dist

@jit(nopython=True, cache=True)
def compute_progress_jit(x, y, track):
    """進行度計算（JIT最適化）"""
    min_dist = 1e9
    best_idx = 0
    for i in range(len(track)):
        d = math.sqrt((x - track[i, 0])**2 + (y - track[i, 1])**2)
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx / len(track)

@jit(nopython=True, cache=True)
def sense_jit(x, y, theta, track, half_width, angles, max_dist):
    """センサー読み取り（JIT最適化）"""
    readings = np.zeros(len(angles))
    for idx in range(len(angles)):
        angle = theta + angles[idx]
        dx = math.cos(angle)
        dy = math.sin(angle)
        
        d = 0.0
        step = 3.0
        hit = False
        
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
    """ニューラルネットワーク順伝播（JIT最適化）"""
    p = 0
    hidden = np.zeros(nhid)
    
    # 入力→隠れ層
    for j in range(nhid):
        s = 0.0
        for i in range(nin):
            s += weights[p] * inputs[i]
            p += 1
        s += weights[p]  # バイアス
        p += 1
        hidden[j] = math.tanh(s)
    
    # 隠れ→出力層
    output = np.zeros(nout)
    for k in range(nout):
        s = 0.0
        for j in range(nhid):
            s += weights[p] * hidden[j]
            p += 1
        s += weights[p]  # バイアス
        p += 1
        output[k] = math.tanh(s)
    
    return output


@jit(nopython=True, cache=True)
def simulate_car_jit(weights, track, half_width, sensor_angles, 
                     sim_steps, dt, max_speed, max_steer, throttle_power, wheelbase):
    """車両シミュレーション＋複合評価（進捗 + 速度 - 逆走ペナルティ）"""
    x, y = track[0, 0], track[0, 1]
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0

    max_progress = 0.0
    last_progress = 0.0
    wrong_dir_amount = 0.0  # 逆走量（進捗の後退分）
    sum_forward_speed = 0.0
    alive_steps = 0

    for step in range(sim_steps):
        # センサー読み取り
        sensor_readings = sense_jit(x, y, theta, track, half_width, 
                                    sensor_angles, MAX_SENSOR_DIST)

        # NN入力
        nn_input = np.empty(NIN)
        for i in range(NSENS):
            nn_input[i] = sensor_readings[i]
        nn_input[NSENS] = v / max_speed

        # NN出力
        outputs = nn_forward_jit(weights, nn_input, NIN, NHID, NOUT)
        steer = max(-1.0, min(1.0, outputs[0])) * max_steer
        throttle = max(-1.0, min(1.0, outputs[1]))

        # 車両運動
        v += throttle * throttle_power * dt
        v = max(-1.0, min(max_speed, v))
        theta += (v * math.tan(steer) / wheelbase) * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt

        # コースアウト判定
        dist_to_center = distance_to_track_jit(x, y, track)
        if dist_to_center > half_width:
            break

        # 進捗
        progress = compute_progress_jit(x, y, track)

        # 逆走判定（進捗の後退分をペナルティとして蓄積）
        delta = progress - last_progress
        if delta >= 0.0:
            if progress > max_progress:
                max_progress = progress
        else:
            wrong_dir_amount += -delta  # 後退した分だけ増やす

        last_progress = progress

        # 速度（前進成分のみ加算）
        if v > 0.0:
            sum_forward_speed += v

        alive_steps += 1

    # 平均速度を正規化
    if alive_steps > 0:
        avg_speed = sum_forward_speed / alive_steps
    else:
        avg_speed = 0.0

    avg_speed_norm = avg_speed / max_speed  # 0〜1程度

    # 重み（必要に応じて調整）
    w_prog = 0.7
    w_speed = 0.3
    w_wrong = 2.0

    fitness = w_prog * max_progress + w_speed * avg_speed_norm - w_wrong * wrong_dir_amount

    if fitness < 0.0:
        fitness = 0.0

    return fitness

# ==========================================
# DEAP遺伝的アルゴリズム
# ==========================================
def eval_individual(individual):
    """個体評価関数（マルチプロセス用）"""
    weights = np.array(individual)
    fitness = simulate_car_jit(weights, TRACK, HALF_WIDTH, SENSOR_ANGLES,
                              SIM_STEPS, DT, MAX_SPEED, MAX_STEER, 
                              THROTTLE_POWER, WHEELBASE)
    return (fitness,)

def init_deap():
    """DEAPのセットアップ"""
    # 型定義
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # 遺伝子と個体の生成
    toolbox.register("attr_float", np.random.normal, 0, 0.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                    toolbox.attr_float, n=N_WEIGHTS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 遺伝的操作
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

def evolve_with_deap(track, half_width):
    """DEAPで進化"""
    global TRACK, HALF_WIDTH
    TRACK = track
    HALF_WIDTH = half_width
    
    print(f"Neural Network: {NIN} inputs -> {NHID} hidden -> {NOUT} outputs")
    print(f"Total weights: {N_WEIGHTS}")
    print(f"Population: {POP_SIZE}, Generations: {GENERATIONS}")
    print(f"Track points: {len(track)}, Half width: {half_width}")
    print(f"CPU cores: {cpu_count()}")
    print()
    
    toolbox = init_deap()
    
    # マルチプロセスプールの登録
    pool = Pool(processes=cpu_count())
    toolbox.register("map", pool.map)
    
    # 初期集団
    pop = toolbox.population(n=POP_SIZE)
    
    # 統計
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    
    # 殿堂（ベスト保存）
    hof = tools.HallOfFame(1)
    
    # 進化アルゴリズム実行
    pop, logbook = algorithms.eaSimple(pop, toolbox, 
                                       cxpb=CXPB, mutpb=MUTPB, 
                                       ngen=GENERATIONS,
                                       stats=stats, halloffame=hof,
                                       verbose=True)
    
    pool.close()
    pool.join()
    
    # ベスト個体
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]
    
    print("\n✅ Evolution complete!")
    print(f"Best fitness: {best_fitness:.4f} (progress+speed-penalty)")

    
    return np.array(best_individual), pop

# ==========================================
# 可視化用シミュレーション
# ==========================================
def simulate_for_visualization(weights, track, half_width):
    """可視化用の軌跡生成"""
    x, y = track[0]
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0
    trajectory = [(float(x), float(y))]
    
    for step in range(SIM_STEPS):
        sensor_readings = sense_jit(x, y, theta, track, half_width,
                                    SENSOR_ANGLES, MAX_SENSOR_DIST)
        nn_input = np.append(sensor_readings, v / MAX_SPEED)
        outputs = nn_forward_jit(weights, nn_input, NIN, NHID, NOUT)
        
        steer = np.clip(outputs[0], -1.0, 1.0) * MAX_STEER
        throttle = np.clip(outputs[1], -1.0, 1.0)
        
        v += throttle * THROTTLE_POWER * DT
        v = np.clip(v, -1.0, MAX_SPEED)
        theta += (v * math.tan(steer) / WHEELBASE) * DT
        x += v * math.cos(theta) * DT
        y += v * math.sin(theta) * DT
        
        trajectory.append((float(x), float(y)))
        
        if distance_to_track_jit(x, y, track) > half_width:
            break
    
    return trajectory

def visualize_results(track, half_width, best_weights, population):
    """結果可視化"""
    print("\nGenerating visualization...")
    
    # ベスト軌跡
    best_trajectory = simulate_for_visualization(best_weights, track, half_width)
    
    # サンプル軌跡
    sample_trajectories = []
    sample_size = min(30, len(population))
    sample_indices = np.random.choice(len(population), sample_size, replace=False)
    
    for idx in sample_indices:
        weights = np.array(population[idx])
        traj = simulate_for_visualization(weights, track, half_width)
        sample_trajectories.append(traj)
    
    # プロット
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('#000000')
    fig.patch.set_facecolor('#000000')
    
    # トラック
    tx, ty = track[:, 0], track[:, 1]
    ax.plot(tx, ty, color='#333333', linewidth=2, alpha=0.6, label='Track Center')
    
    for i in range(0, len(track), 10):
        c = plt.Circle(track[i], half_width, color='#1a1a1a', alpha=0.15, fill=True)
        ax.add_artist(c)
    
    # サンプル軌跡
    for traj in sample_trajectories:
        if len(traj) > 1:
            px, py = zip(*traj)
            ax.plot(px, py, color='#00ffff', linewidth=0.5, alpha=0.2)
    
    # ベスト軌跡
    if len(best_trajectory) > 1:
        bx, by = zip(*best_trajectory)
        ax.plot(bx, by, color='#ffaa00', linewidth=4.5, alpha=1.0,
               label=f'Best AI ({len(best_trajectory)} steps)', zorder=10)
        ax.plot(bx[0], by[0], 'o', color='#00ff00', markersize=18,
               label='Start', zorder=11, markeredgecolor='white', markeredgewidth=2.5)
        ax.plot(bx[-1], by[-1], 'o', color='#ff0000', markersize=18,
               label='End', zorder=11, markeredgecolor='white', markeredgewidth=2.5)
    
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', facecolor='#0a0a0a', edgecolor='#00ffff',
             labelcolor='white', fontsize=12, framealpha=0.95)
    
    title = '🚀 Ultra-Fast GA Evolution (DEAP'
    if NUMBA_AVAILABLE:
        title += ' + Numba JIT'
    title += ' + Multiprocessing)'
    
    ax.set_title(title, color='white', fontsize=18, pad=25, fontweight='bold')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}.png"
    plt.savefig(output_filename, facecolor='#000000', dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {output_filename}")
    
    plt.show()

# ==========================================
# メイン
# ==========================================
def main():
    print("=" * 75)
    print("  🚀 Ultra-Fast Genetic Algorithm - Neural Network Evolution")
    print(f"  Libraries: DEAP + {'Numba JIT + ' if NUMBA_AVAILABLE else ''}Multiprocessing")
    print("=" * 75)
    print()
    
    # トラック生成
    print("Generating track...")
    track, half_width = generate_track()
    print(f"Track: {len(track)} points\n")
    
    # 進化
    start_time = datetime.now()
    best_weights, final_pop = evolve_with_deap(track, half_width)
    end_time = datetime.now()
    
    elapsed = (end_time - start_time).total_seconds()
    print(f"\n⏱️  Time: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"⚡ Speed: {GENERATIONS/elapsed:.2f} generations/sec")
    
    # 可視化
    visualize_results(track, half_width, best_weights, final_pop)

if __name__ == '__main__':
    main()
