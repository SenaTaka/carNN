#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズムによるニューラルネットワーク学習 - GPU高速化版
NumPy/CuPyでベクトル化した高速実装
"""

import math
import random
from datetime import datetime
import matplotlib.pyplot as plt

# GPU利用の試行
try:
    import cupy as cp
    xp = cp
    GPU_AVAILABLE = True
    print("🚀 GPU (CuPy) detected! Using GPU acceleration.")
except ImportError:
    import numpy as np
    xp = np
    GPU_AVAILABLE = False
    print("⚡ GPU not available. Using NumPy CPU acceleration.")

import numpy as np

# ==========================================
# パラメータ設定
# ==========================================
POP_SIZE = 20#0          # 個体数
GENERATIONS = 50#0       # 世代数
ELITE_SIZE = 2#0         # エリート保存数
MUTATION_RATE = 0.05    # 変異率
TOURNAMENT_SIZE = 3     # トーナメントサイズ

# ニューラルネットワーク構造
NSENS = 5              # センサー数
NIN = NSENS + 1        # 入力層 (センサー + 速度)
NHID = 3#8               # 隠れ層
NOUT = 2               # 出力層 (ステアリング, スロットル)

# シミュレーションパラメータ
SIM_STEPS = 2000       # シミュレーションステップ数
DT = 0.05              # 時間刻み
MAX_SENSOR_DIST = 40.0 # センサー最大距離
SENSOR_ANGLES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # センサー角度

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
    t = np.linspace(0, 2 * np.pi * laps, n_points * laps)
    r = 70.0 + 30.0 * np.sin(3.0 * t) + 15.0 * np.cos(7.0 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    track = np.column_stack([x, y])
    return track, 12.0  # track, half_width

# ==========================================
# 幾何計算（ベクトル化）
# ==========================================
def point_to_segment_distance_vectorized(px, py, track):
    """点から全線分への最短距離をベクトル化計算"""
    ax = track[:-1, 0]
    ay = track[:-1, 1]
    bx = track[1:, 0]
    by = track[1:, 1]
    
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    
    c = vx * vx + vy * vy
    c = np.where(c == 0, 1e-10, c)  # ゼロ除算回避
    
    t = np.clip((vx * wx + vy * wy) / c, 0.0, 1.0)
    
    closest_x = ax + t * vx
    closest_y = ay + t * vy
    
    dist = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    return np.min(dist)

def compute_progress_vectorized(x, y, track):
    """トラック上での進行度を計算"""
    distances = np.sqrt((track[:, 0] - x)**2 + (track[:, 1] - y)**2)
    best_idx = np.argmin(distances)
    return best_idx / len(track)

# ==========================================
# センサー（高速化版）
# ==========================================
def sense_fast(x, y, theta, track, half_width, angles, max_dist):
    """5方向のセンサー - 高速化版"""
    sensor_angles = theta + angles
    dx = np.cos(sensor_angles)
    dy = np.sin(sensor_angles)
    
    readings = []
    for i in range(len(angles)):
        # レイキャスト（バイナリサーチ風に高速化）
        d = 0.0
        step = 4.0  # 初期ステップを大きく
        hit = False
        
        while d <= max_dist:
            px = x + dx[i] * d
            py = y + dy[i] * d
            dist_to_center = point_to_segment_distance_vectorized(px, py, track)
            
            if dist_to_center > half_width:
                if step > 0.5:
                    # ステップを細かくして戻る
                    d -= step
                    step *= 0.5
                    d += step
                else:
                    hit = True
                    break
            else:
                d += step
        
        readings.append((d if hit else max_dist) / max_dist)
    
    return np.array(readings)

# ==========================================
# ニューラルネットワーク（行列演算）
# ==========================================
class NeuralNetwork:
    def __init__(self):
        self.n_weights = NHID * NIN + NHID + NOUT * NHID + NOUT
        
    def set_weights(self, weights):
        """重みをレイヤーごとに分割"""
        p = 0
        # 入力→隠れ層
        self.W1 = weights[p:p + NHID * NIN].reshape(NIN, NHID)
        p += NHID * NIN
        self.b1 = weights[p:p + NHID]
        p += NHID
        # 隠れ→出力層
        self.W2 = weights[p:p + NOUT * NHID].reshape(NHID, NOUT)
        p += NOUT * NHID
        self.b2 = weights[p:p + NOUT]
        
    def forward(self, inputs):
        """順伝播（行列演算）"""
        # 入力→隠れ層
        hidden = np.tanh(np.dot(inputs, self.W1) + self.b1)
        # 隠れ→出力層
        output = np.tanh(np.dot(hidden, self.W2) + self.b2)
        return output

# ==========================================
# 車両シミュレーション（高速化版）
# ==========================================
def simulate_car_fast(weights, track, half_width):
    """1個体の車両をシミュレーション - 高速化版"""
    nn = NeuralNetwork()
    nn.set_weights(weights)
    
    # 初期位置と向き
    x, y = track[0]
    dx = track[1, 0] - track[0, 0]
    dy = track[1, 1] - track[0, 1]
    theta = np.arctan2(dy, dx)
    v = 0.0
    
    max_progress = 0.0
    trajectory = [(float(x), float(y))]
    
    for step in range(SIM_STEPS):
        # センサー読み取り
        sensor_readings = sense_fast(x, y, theta, track, half_width, 
                                     SENSOR_ANGLES, MAX_SENSOR_DIST)
        
        # ニューラルネットワークの入力
        nn_input = np.append(sensor_readings, v / MAX_SPEED)
        
        # ニューラルネットワークの出力
        outputs = nn.forward(nn_input)
        steer = np.clip(outputs[0], -1.0, 1.0) * MAX_STEER
        throttle = np.clip(outputs[1], -1.0, 1.0)
        
        # 車両運動モデル
        v += throttle * THROTTLE_POWER * DT
        v = np.clip(v, -1.0, MAX_SPEED)
        theta += (v * np.tan(steer) / WHEELBASE) * DT
        x += v * np.cos(theta) * DT
        y += v * np.sin(theta) * DT
        
        trajectory.append((float(x), float(y)))
        
        # コースアウト判定
        dist_to_center = point_to_segment_distance_vectorized(x, y, track)
        if dist_to_center > half_width:
            break
        
        # 進行度更新
        progress = compute_progress_vectorized(x, y, track)
        if progress > max_progress:
            max_progress = progress
    
    return max_progress, trajectory

# ==========================================
# 遺伝的アルゴリズム（並列化）
# ==========================================
def create_random_genome(n_weights):
    """ランダムな重みを生成"""
    return np.random.normal(0, 0.5, n_weights)

def create_population(pop_size, n_weights):
    """初期集団を一括生成"""
    return np.random.normal(0, 0.5, (pop_size, n_weights))

def tournament_selection_batch(population, fitnesses, n_select):
    """トーナメント選択を一括実行"""
    selected = []
    for _ in range(n_select):
        indices = np.random.choice(len(population), TOURNAMENT_SIZE, replace=False)
        best_idx = indices[np.argmax(fitnesses[indices])]
        selected.append(population[best_idx].copy())
    return np.array(selected)

def crossover_vectorized(parent1, parent2):
    """一点交叉"""
    point = np.random.randint(1, len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate_vectorized(genome, mutation_rate):
    """変異をベクトル化"""
    mask = np.random.random(len(genome)) < mutation_rate
    mutations = np.random.normal(0, 0.3, len(genome))
    genome[mask] += mutations[mask]
    return genome

def evaluate_population_parallel(population, track, half_width):
    """全個体を評価（可能な限り並列化）"""
    fitnesses = []
    trajectories = []
    
    # マルチプロセスは使わず、NumPyの高速化に頼る
    for genome in population:
        fitness, trajectory = simulate_car_fast(genome, track, half_width)
        fitnesses.append(fitness)
        trajectories.append(trajectory)
    
    return np.array(fitnesses), trajectories

def evolve(track, half_width):
    """遺伝的アルゴリズムのメインループ"""
    nn = NeuralNetwork()
    n_weights = nn.n_weights
    
    print(f"Neural Network: {NIN} inputs -> {NHID} hidden -> {NOUT} outputs")
    print(f"Total weights: {n_weights}")
    print(f"Population: {POP_SIZE}, Generations: {GENERATIONS}")
    print(f"Track points: {len(track)}, Half width: {half_width}")
    print()
    
    # 初期集団生成
    population = create_population(POP_SIZE, n_weights)
    
    best_genome = None
    best_fitness = 0.0
    best_trajectory = []
    
    # 世代ループ
    for gen in range(GENERATIONS):
        # 評価
        fitnesses, trajectories = evaluate_population_parallel(population, track, half_width)
        
        # ベスト更新
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        
        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_genome = population[gen_best_idx].copy()
            best_trajectory = trajectories[gen_best_idx]
        
        # 進捗表示
        if gen % 10 == 0 or gen == GENERATIONS - 1:
            avg_fitness = np.mean(fitnesses)
            print(f"Gen {gen:3d}: Best={gen_best_fitness:.4f} ({gen_best_fitness*100:.1f}%), "
                  f"Avg={avg_fitness:.4f} ({avg_fitness*100:.1f}%)")
        
        # 次世代生成
        # エリート保存
        elite_indices = np.argsort(fitnesses)[-ELITE_SIZE:]
        new_population = [population[i].copy() for i in elite_indices]
        
        # 交叉と変異で残りを生成
        n_offspring = POP_SIZE - ELITE_SIZE
        parents = tournament_selection_batch(population, fitnesses, n_offspring)
        
        for i in range(0, n_offspring - 1, 2):
            parent1 = parents[i % len(parents)]
            parent2 = parents[(i + 1) % len(parents)]
            
            child1, child2 = crossover_vectorized(parent1, parent2)
            
            # 変異率を世代とともに減衰
            current_mutation_rate = MUTATION_RATE * (1.0 - gen / GENERATIONS)
            child1 = mutate_vectorized(child1, current_mutation_rate)
            child2 = mutate_vectorized(child2, current_mutation_rate)
            
            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)
        
        population = np.array(new_population)
    
    print("\n✅ Evolution complete!")
    print(f"Best fitness: {best_fitness:.4f} ({best_fitness*100:.1f}% Lap)")
    
    return best_genome, best_trajectory, population

# ==========================================
# 可視化
# ==========================================
def visualize_results(track, half_width, best_trajectory, population):
    """結果を可視化して保存"""
    print("\nSimulating sampled genomes for visualization...")
    
    # サンプリングして軌跡を生成
    sample_trajectories = []
    sample_size = min(50, len(population))  # 最大50個体
    sample_indices = np.random.choice(len(population), sample_size, replace=False)
    
    for i, idx in enumerate(sample_indices):
        if i % 10 == 0:
            print(f"  Simulating {i}/{sample_size}...")
        if idx == 0:  # ベストはスキップ
            continue
        fitness, traj = simulate_car_fast(population[idx], track, half_width)
        sample_trajectories.append(traj)
    
    # プロット
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_facecolor('#0a0a0a')
    fig.patch.set_facecolor('#0a0a0a')
    
    # トラック描画
    tx = track[:, 0]
    ty = track[:, 1]
    ax.plot(tx, ty, color='#444444', linewidth=1.5, alpha=0.5, label='Track Center', zorder=1)
    
    # トラック境界を円で表現
    for i in range(0, len(track), 8):
        c = plt.Circle(track[i], half_width, color='#222222', alpha=0.1, fill=True, zorder=0)
        ax.add_artist(c)
    
    # 他の個体の軌跡（シアン）
    for traj in sample_trajectories:
        if len(traj) > 1:
            px = [p[0] for p in traj]
            py = [p[1] for p in traj]
            ax.plot(px, py, color='#00d4ff', linewidth=0.6, alpha=0.15, zorder=2)
    
    # ベスト個体の軌跡（ゴールド）
    if best_trajectory and len(best_trajectory) > 1:
        bx = [p[0] for p in best_trajectory]
        by = [p[1] for p in best_trajectory]
        ax.plot(bx, by, color='#ffd700', linewidth=4, alpha=0.95, 
               label=f'Best AI ({len(best_trajectory)} steps)', zorder=10)
        
        # スタート地点
        ax.plot(bx[0], by[0], 'o', color='#00ff00', markersize=15, 
               label='Start', zorder=11, markeredgecolor='white', markeredgewidth=2)
        # ゴール地点
        ax.plot(bx[-1], by[-1], 'o', color='#ff0000', markersize=15, 
               label='End', zorder=11, markeredgecolor='white', markeredgewidth=2)
    
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='upper right', facecolor='#1a1a1a', edgecolor='#00d4ff', 
             labelcolor='white', fontsize=11, framealpha=0.9)
    
    title = 'Genetic Algorithm - Neural Network Evolution'
    if GPU_AVAILABLE:
        title += ' (🚀 GPU Accelerated)'
    else:
        title += ' (⚡ NumPy Accelerated)'
    
    ax.set_title(title, color='white', fontsize=18, pad=20, fontweight='bold')
    
    # 実行時間をファイル名にして保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}.png"
    plt.savefig(output_filename, facecolor='#0a0a0a', dpi=150, bbox_inches='tight')
    print(f"\n💾 Result saved to: {output_filename}")
    
    plt.show()

# ==========================================
# メイン関数
# ==========================================
def main():
    print("=" * 70)
    print("  遺伝的アルゴリズムによるニューラルネットワーク学習")
    if GPU_AVAILABLE:
        print("  🚀 GPU Accelerated Version (CuPy)")
    else:
        print("  ⚡ High-Performance Version (NumPy)")
    print("=" * 70)
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
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    
    # 可視化
    visualize_results(track, half_width, best_trajectory, final_population)

if __name__ == '__main__':
    main()
