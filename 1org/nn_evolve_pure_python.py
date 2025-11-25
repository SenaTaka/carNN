#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズムによるニューラルネットワーク学習 - Pure Python版
C言語版の機能を完全にPythonで実装
"""

import math
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# パラメータ設定
# ==========================================
POP_SIZE = 200          # 個体数
GENERATIONS = 500       # 世代数
ELITE_SIZE = 20         # エリート保存数
MUTATION_RATE = 0.05    # 変異率
TOURNAMENT_SIZE = 3     # トーナメントサイズ

# ニューラルネットワーク構造
NSENS = 5              # センサー数
NIN = NSENS + 1        # 入力層 (センサー + 速度)
NHID = 8               # 隠れ層
NOUT = 2               # 出力層 (ステアリング, スロットル)

# シミュレーションパラメータ
SIM_STEPS = 2000       # シミュレーションステップ数
DT = 0.05              # 時間刻み
MAX_SENSOR_DIST = 40.0 # センサー最大距離
SENSOR_ANGLES = [-1.0, -0.5, 0.0, 0.5, 1.0]  # センサー角度

# 車両パラメータ
WHEELBASE = 0.5        # ホイールベース
MAX_SPEED = 8.0        # 最高速度
MAX_STEER = 0.7        # 最大ステアリング角
THROTTLE_POWER = 3.5   # スロットルパワー

# ==========================================
# コース生成
# ==========================================
def generate_track(n_points=200, laps=2):
    """複雑な形状のサーキットコースを生成"""
    pts = []
    for i in range(n_points * laps):
        t = (i / n_points) * 2 * math.pi
        r = 70.0 + 30.0 * math.sin(3.0 * t) + 15.0 * math.cos(7.0 * t)
        x = r * math.cos(t)
        y = r * math.sin(t)
        pts.append((x, y))
    return pts, 12.0  # track, half_width

# ==========================================
# 幾何計算
# ==========================================
def point_to_segment_distance(px, py, ax, ay, bx, by):
    """点から線分への最短距離"""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c = vx * vx + vy * vy
    if c == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (vx * wx + vy * wy) / c))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))

def distance_to_track(x, y, track):
    """車両位置からトラック中心線への最短距離"""
    min_dist = float('inf')
    for i in range(len(track) - 1):
        d = point_to_segment_distance(x, y, track[i][0], track[i][1], 
                                      track[i+1][0], track[i+1][1])
        if d < min_dist:
            min_dist = d
    return min_dist

def compute_progress(x, y, track):
    """トラック上での進行度を計算 (0.0 ~ 1.0)"""
    min_dist = float('inf')
    best_idx = 0
    for i in range(len(track)):
        d = math.hypot(x - track[i][0], y - track[i][1])
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx / len(track)

# ==========================================
# センサー
# ==========================================
def sense(x, y, theta, track, half_width, angles, max_dist):
    """5方向のセンサーで壁までの距離を測定"""
    readings = []
    for angle in angles:
        sensor_angle = theta + angle
        dx = math.cos(sensor_angle)
        dy = math.sin(sensor_angle)
        
        # レイキャスト
        d = 0.0
        step = 2.0
        hit = False
        while d <= max_dist:
            px = x + dx * d
            py = y + dy * d
            dist_to_center = distance_to_track(px, py, track)
            if dist_to_center > half_width:
                hit = True
                break
            d += step
        
        readings.append((d if hit else max_dist) / max_dist)
    return readings

# ==========================================
# ニューラルネットワーク
# ==========================================
def calculate_weights_count():
    """重みの総数を計算"""
    return NHID * NIN + NHID + NOUT * NHID + NOUT

def nn_forward(weights, inputs):
    """順伝播計算"""
    p = 0
    # 隠れ層
    hidden = []
    for j in range(NHID):
        s = 0.0
        for i in range(NIN):
            s += weights[p] * inputs[i]
            p += 1
        s += weights[p]  # バイアス
        p += 1
        hidden.append(math.tanh(s))
    
    # 出力層
    outputs = []
    for k in range(NOUT):
        s = 0.0
        for j in range(NHID):
            s += weights[p] * hidden[j]
            p += 1
        s += weights[p]  # バイアス
        p += 1
        outputs.append(math.tanh(s))
    
    return outputs

# ==========================================
# 車両シミュレーション
# ==========================================
def simulate_car(weights, track, half_width):
    """1個体の車両をシミュレーション"""
    # 初期位置と向き
    x, y = track[0]
    theta = math.atan2(track[1][1] - y, track[1][0] - x)
    v = 0.0  # 速度
    
    max_progress = 0.0
    trajectory = [(x, y)]
    
    for step in range(SIM_STEPS):
        # センサー読み取り
        sensor_readings = sense(x, y, theta, track, half_width, 
                               SENSOR_ANGLES, MAX_SENSOR_DIST)
        
        # ニューラルネットワークの入力
        nn_input = sensor_readings + [v / MAX_SPEED]
        
        # ニューラルネットワークの出力
        outputs = nn_forward(weights, nn_input)
        steer = max(-1.0, min(1.0, outputs[0])) * MAX_STEER
        throttle = max(-1.0, min(1.0, outputs[1]))
        
        # 車両運動モデル (キネマティックバイシクルモデル)
        v += throttle * THROTTLE_POWER * DT
        v = max(-1.0, min(MAX_SPEED, v))
        theta += (v * math.tan(steer) / WHEELBASE) * DT
        x += v * math.cos(theta) * DT
        y += v * math.sin(theta) * DT
        
        trajectory.append((x, y))
        
        # コースアウト判定
        dist_to_center = distance_to_track(x, y, track)
        if dist_to_center > half_width:
            break
        
        # 進行度更新
        progress = compute_progress(x, y, track)
        if progress > max_progress:
            max_progress = progress
    
    return max_progress, trajectory

# ==========================================
# 遺伝的アルゴリズム
# ==========================================
def create_random_genome(n_weights):
    """ランダムな重みを生成"""
    return [random.gauss(0, 0.5) for _ in range(n_weights)]

def tournament_selection(population, fitnesses):
    """トーナメント選択"""
    selected_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
    best_idx = max(selected_indices, key=lambda i: fitnesses[i])
    return population[best_idx][:]

def crossover(parent1, parent2):
    """一点交叉"""
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(genome, mutation_rate):
    """変異"""
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] += random.gauss(0, 0.3)
    return genome

def evaluate_population(population, track, half_width):
    """全個体を評価"""
    fitnesses = []
    trajectories = []
    for genome in population:
        fitness, trajectory = simulate_car(genome, track, half_width)
        fitnesses.append(fitness)
        trajectories.append(trajectory)
    return fitnesses, trajectories

def evolve(track, half_width):
    """遺伝的アルゴリズムのメインループ"""
    n_weights = calculate_weights_count()
    print(f"Neural Network: {NIN} inputs -> {NHID} hidden -> {NOUT} outputs")
    print(f"Total weights: {n_weights}")
    print(f"Population: {POP_SIZE}, Generations: {GENERATIONS}")
    print(f"Track points: {len(track)}, Half width: {half_width}")
    print()
    
    # 初期集団生成
    population = [create_random_genome(n_weights) for _ in range(POP_SIZE)]
    
    best_genome = None
    best_fitness = 0.0
    best_trajectory = []
    
    # 世代ループ
    for gen in range(GENERATIONS):
        # 評価
        fitnesses, trajectories = evaluate_population(population, track, half_width)
        
        # ベスト更新
        gen_best_idx = fitnesses.index(max(fitnesses))
        gen_best_fitness = fitnesses[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = population[gen_best_idx][:]
            best_trajectory = trajectories[gen_best_idx]
        
        # 進捗表示
        if gen % 10 == 0 or gen == GENERATIONS - 1:
            print(f"Gen {gen:3d}: Best Fitness = {gen_best_fitness:.4f} ({gen_best_fitness*100:.1f}% Lap)")
        
        # 次世代生成
        # エリート保存
        elite_indices = sorted(range(len(fitnesses)), 
                              key=lambda i: fitnesses[i], 
                              reverse=True)[:ELITE_SIZE]
        new_population = [population[i][:] for i in elite_indices]
        
        # 交叉と変異で残りを生成
        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            
            # 変異率を世代とともに減衰
            current_mutation_rate = MUTATION_RATE * (1.0 - gen / GENERATIONS)
            child1 = mutate(child1, current_mutation_rate)
            child2 = mutate(child2, current_mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)
        
        population = new_population
    
    print("\nEvolution complete!")
    print(f"Best fitness: {best_fitness:.4f} ({best_fitness*100:.1f}% Lap)")
    
    return best_genome, best_trajectory, population

# ==========================================
# 可視化
# ==========================================
def visualize_results(track, half_width, best_trajectory, population):
    """結果を可視化して保存"""
    print("\nSimulating all genomes for visualization...")
    
    # サンプリングして軌跡を生成（全個体は重いので間引く）
    sample_trajectories = []
    for i in range(0, len(population), 3):  # 3個体に1個
        if i == 0:  # ベストはスキップ
            continue
        fitness, traj = simulate_car(population[i], track, half_width)
        sample_trajectories.append(traj)
    
    # プロット
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # トラック描画
    tx = [p[0] for p in track]
    ty = [p[1] for p in track]
    ax.plot(tx, ty, color='white', linewidth=1, alpha=0.3, label='Track Center')
    
    # トラック境界を円で表現
    for i in range(0, len(track), 5):
        c = plt.Circle(track[i], half_width, color='white', alpha=0.05, fill=True)
        ax.add_artist(c)
    
    # 他の個体の軌跡（シアン）
    for traj in sample_trajectories:
        if len(traj) > 1:
            px = [p[0] for p in traj]
            py = [p[1] for p in traj]
            ax.plot(px, py, color='cyan', linewidth=0.8, alpha=0.2)
    
    # ベスト個体の軌跡（赤）
    if best_trajectory and len(best_trajectory) > 1:
        bx = [p[0] for p in best_trajectory]
        by = [p[1] for p in best_trajectory]
        ax.plot(bx, by, color='red', linewidth=3, alpha=0.9, label='Best AI', zorder=10)
        
        # スタート地点
        ax.plot(bx[0], by[0], 'go', markersize=12, label='Start', zorder=11)
        # ゴール地点
        ax.plot(bx[-1], by[-1], 'ro', markersize=12, label='End', zorder=11)
    
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', facecolor='black', edgecolor='white', 
             labelcolor='white', fontsize=10)
    ax.set_title('Genetic Algorithm - Neural Network Evolution', 
                color='white', fontsize=16, pad=20)
    
    # 実行時間をファイル名にして保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}.png"
    plt.savefig(output_filename, facecolor='black', dpi=150, bbox_inches='tight')
    print(f"\nResult saved to: {output_filename}")
    
    plt.show()

# ==========================================
# メイン関数
# ==========================================
def main():
    print("=" * 60)
    print("遺伝的アルゴリズムによるニューラルネットワーク学習")
    print("Pure Python Implementation")
    print("=" * 60)
    print()
    
    # コース生成
    print("Generating track...")
    track, half_width = generate_track()
    print(f"Track generated: {len(track)} points\n")
    
    # 進化
    start_time = datetime.now()
    best_genome, best_trajectory, final_population = evolve(track, half_width)
    end_time = datetime.now()
    
    elapsed = (end_time - start_time).total_seconds()
    print(f"\nTotal time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    # 可視化
    visualize_results(track, half_width, best_trajectory, final_population)

if __name__ == '__main__':
    main()
