#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズムによるニューラルネットワーク学習 - 超高速改良版
DEAP + Numba + マルチプロセスで最大限高速化
指定方向（track[0] → track[1] → ...）のみを前進として評価

改良点:
- 適応的な突然変異率（世代に応じて減衰）
- エリート保存戦略の実装
- チェックポイントシステム（自動保存/読み込み）
- より複雑なニューラルネットワーク（ReLU活性化関数オプション）
- 改善された評価関数（スムーズさ、衝突ペナルティ）
- リアルタイム進捗グラフ
- 複数コースバリエーション
"""

import math
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
ELITE_SIZE = 10  # エリート個体数
CXPB = 0.7  # 交叉確率
MUTPB_INITIAL = 0.3  # 初期変異確率
MUTPB_FINAL = 0.1    # 最終変異確率

NSENS = 5
NIN = NSENS + 1
NHID = 12  # 隠れ層を8→12に増加
NOUT = 2

SIM_STEPS = 2000
DT = 0.05
MAX_SENSOR_DIST = 40.0
SENSOR_ANGLES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

WHEELBASE = 0.5
MAX_SPEED = 8.0
MAX_STEER = 0.7
THROTTLE_POWER = 3.5

# チェックポイント設定
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 50  # 50世代ごとに保存
REALTIME_UPDATE_INTERVAL = 10  # リアルタイム表示の更新間隔（世代）

# グローバル変数（マルチプロセスで共有）
TRACK = None
HALF_WIDTH = None
TRACK_LAPS = 2  # トラックのラップ数（進捗率計算に使用）
N_WEIGHTS = NHID * NIN + NHID + NOUT * NHID + NOUT

# ==========================================
# コース生成
# ==========================================
def generate_track(track_type="circuit", n_points=200, laps=2):
    """
    複数タイプのコース生成
    
    track_type: "circuit" (円形サーキット), "oval" (楕円), "figure8" (8の字)
    """
    if track_type == "circuit":
        t = np.linspace(0, 2 * np.pi * laps, n_points * laps)
        r = 70.0 + 30.0 * np.sin(3.0 * t) + 15.0 * np.cos(7.0 * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y]), 12.0
    
    elif track_type == "oval":
        t = np.linspace(0, 2 * np.pi * laps, n_points * laps)
        a, b = 100.0, 50.0  # 楕円の長軸・短軸
        x = a * np.cos(t) + 20.0 * np.sin(5.0 * t)
        y = b * np.sin(t) + 10.0 * np.cos(5.0 * t)
        return np.column_stack([x, y]), 15.0
    
    elif track_type == "figure8":
        t = np.linspace(0, 4 * np.pi * laps, n_points * laps)
        r = 60.0
        x = r * np.sin(t)
        y = r * np.sin(t) * np.cos(t)
        return np.column_stack([x, y]), 10.0
    
    else:
        # デフォルトはcircuit
        return generate_track("circuit", n_points, laps)

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
        dx = px - ax
        dy = py - ay
        return math.sqrt(dx * dx + dy * dy)
    t = (vx * wx + vy * wy) / c
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    closest_x = ax + t * vx
    closest_y = ay + t * vy
    dx = px - closest_x
    dy = py - closest_y
    return math.sqrt(dx * dx + dy * dy)

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
def nearest_track_index_jit(x, y, track):
    """トラック上で最も近い点のインデックス（JIT最適化）"""
    min_dist = 1e18
    best_idx = 0
    for i in range(len(track)):
        dx = x - track[i, 0]
        dy = y - track[i, 1]
        d = dx * dx + dy * dy
        if d < min_dist:
            min_dist = d
            best_idx = i
    return best_idx

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
    """ニューラルネットワーク順伝播（JIT最適化、ReLU使用）"""
    p = 0
    hidden = np.zeros(nhid)

    # 入力→隠れ層（ReLU活性化）
    for j in range(nhid):
        s = 0.0
        for i in range(nin):
            s += weights[p] * inputs[i]
            p += 1
        s += weights[p]  # バイアス
        p += 1
        hidden[j] = max(0.0, s)  # ReLU

    # 隠れ→出力層（tanh）
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
    """
    車両シミュレーション＋複合評価（改良版）
    - コース配列の方向（track[0]→track[1]→...）のみを前進として評価
    - スムーズさの評価追加
    - 衝突までの距離を考慮
    """
    x, y = track[0, 0], track[0, 1]
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0

    n_points = len(track)

    # 進捗管理（指定方向のみ積算）
    last_idx = 0  # 初期位置のインデックス
    max_idx_reached = 0  # 到達した最大インデックス（前進判定用）
    lap_progress = 0.0      # 0〜laps: 指定方向の累積進捗
    max_progress = 0.0      # 到達した最大進捗（逆走しても減らない）
    wrong_dir_amount = 0.0  # 逆走量
    sum_forward_speed = 0.0
    sum_steer_change = 0.0  # ステアリングの変化量（スムーズさ評価）
    last_steer = 0.0
    alive_steps = 0
    min_wall_dist = 1e9     # 壁までの最小距離

    for _ in range(sim_steps):
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

        # スムーズさ評価（ステアリング変化量）
        sum_steer_change += abs(steer - last_steer)
        last_steer = steer

        # 車両運動
        v += throttle * throttle_power * dt
        v = max(-1.0, min(max_speed, v))
        theta += (v * math.tan(steer) / wheelbase) * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt

        # コースアウト判定
        dist_to_center = distance_to_track_jit(x, y, track)
        wall_dist = half_width - dist_to_center
        if wall_dist < min_wall_dist:
            min_wall_dist = wall_dist
        
        if dist_to_center > half_width:
            break

        # トラック上の最寄りインデックス
        idx = nearest_track_index_jit(x, y, track)

        # 進捗更新：到達した最大インデックスを記録
        # ラップ数を考慮した進捗計算
        forward_dist = (idx - max_idx_reached + n_points) % n_points
        
        if forward_dist > 0 and forward_dist < n_points // 2:
            # 前進している
            if idx > max_idx_reached or (idx < n_points // 4 and max_idx_reached > 3 * n_points // 4):
                # 通常の前進、またはラップ境界を超えた
                current_lap = max_progress // 1.0  # 現在何周目か
                lap_progress = current_lap + (idx / n_points)
                
                if lap_progress > max_progress:
                    max_progress = lap_progress
                    max_idx_reached = idx
        # 逆走判定
        elif forward_dist >= n_points // 2:
            backward_dist = (n_points - forward_dist) / n_points
            wrong_dir_amount += backward_dist * 0.1

        alive_steps += 1

        # 速度（前進成分のみ）
        if v > 0.0:
            sum_forward_speed += v

    # 平均速度を正規化
    if alive_steps > 0:
        avg_speed = sum_forward_speed / alive_steps
        avg_steer_change = sum_steer_change / alive_steps
    else:
        avg_speed = 0.0
        avg_steer_change = 0.0

    avg_speed_norm = avg_speed / max_speed  # 0〜1程度
    smoothness = 1.0 / (1.0 + avg_steer_change * 10.0)  # スムーズさ（0〜1）
    safety = max(0.0, min(1.0, min_wall_dist / half_width))  # 安全マージン

    # 重み（調整済み）
    w_prog = 1.0      # 進捗が最重要
    w_speed = 0.3     # 速度
    w_smooth = 0.1    # スムーズさ
    w_safety = 0.05   # 安全性
    w_wrong = 2.5     # 逆走ペナルティ

    # max_progress: 到達した最大進捗（逆走しても減らない、複数周可能）
    # 進捗が1.0を超える場合（複数周）も適切に評価
    fitness = (w_prog * max_progress + 
               w_speed * avg_speed_norm + 
               w_smooth * smoothness + 
               w_safety * safety - 
               w_wrong * wrong_dir_amount)

    if fitness < 0.0:
        fitness = 0.0

    return fitness

# ==========================================
# リアルタイム可視化
# ==========================================
class RealtimeVisualizer:
    """リアルタイムで進化状況を表示"""
    def __init__(self, track, half_width):
        self.track = track
        self.half_width = half_width
        self.fig = None
        self.axes = None
        self.setup_plot()
        
    def setup_plot(self):
        """プロットの初期設定"""
        self.fig = plt.figure(figsize=(18, 6))
        self.fig.patch.set_facecolor('#000000')
        
        # 3つのサブプロット
        self.ax_fitness = plt.subplot(1, 3, 1)
        self.ax_progress = plt.subplot(1, 3, 2)
        self.ax_trajectory = plt.subplot(1, 3, 3)
        
        for ax in [self.ax_fitness, self.ax_progress, self.ax_trajectory]:
            ax.set_facecolor('#0a0a0a')
        
        # フィットネスグラフ設定
        self.ax_fitness.set_xlabel('Generation', color='white', fontsize=10)
        self.ax_fitness.set_ylabel('Fitness', color='white', fontsize=10)
        self.ax_fitness.set_title('📈 Fitness Evolution', color='white', fontsize=12, fontweight='bold')
        self.ax_fitness.grid(True, alpha=0.2, color='#333333')
        self.ax_fitness.tick_params(colors='white', labelsize=8)
        for spine in self.ax_fitness.spines.values():
            spine.set_edgecolor('#00ffff')
        
        # 進捗グラフ設定
        self.ax_progress.set_xlabel('Generation', color='white', fontsize=10)
        self.ax_progress.set_ylabel('Progress (%)', color='white', fontsize=10)
        self.ax_progress.set_title('🎯 Lap Progress', color='white', fontsize=12, fontweight='bold')
        self.ax_progress.grid(True, alpha=0.2, color='#333333')
        self.ax_progress.tick_params(colors='white', labelsize=8)
        # Y軸の範囲を固定（0-110%）で、進捗の変化を見やすく
        self.ax_progress.set_ylim(0, 110)
        for spine in self.ax_progress.spines.values():
            spine.set_edgecolor('#00ffff')
        
        # 軌跡グラフ設定
        self.ax_trajectory.set_aspect('equal')
        self.ax_trajectory.set_xticks([])
        self.ax_trajectory.set_yticks([])
        self.ax_trajectory.set_title('🏁 Best Trajectory', color='white', fontsize=12, fontweight='bold')
        
        # トラックを描画
        tx, ty = self.track[:, 0], self.track[:, 1]
        self.ax_trajectory.plot(tx, ty, color='#333333', linewidth=1.5, alpha=0.6)
        for i in range(0, len(self.track), 15):
            c = plt.Circle(self.track[i], self.half_width, color='#1a1a1a', alpha=0.15, fill=True)
            self.ax_trajectory.add_artist(c)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.1)
    
    def update(self, stats_history, best_weights):
        """グラフを更新"""
        if not stats_history or len(stats_history['gen']) == 0:
            return
        
        # フィットネスグラフ更新
        self.ax_fitness.clear()
        self.ax_fitness.set_facecolor('#0a0a0a')
        gens = stats_history['gen']
        self.ax_fitness.plot(gens, stats_history['max'], color='#00ff00', linewidth=2, label='Max', marker='o', markersize=2)
        self.ax_fitness.plot(gens, stats_history['avg'], color='#ffaa00', linewidth=1.5, label='Avg', alpha=0.8)
        self.ax_fitness.plot(gens, stats_history['min'], color='#ff4444', linewidth=1, label='Min', alpha=0.6)
        self.ax_fitness.set_xlabel('Generation', color='white', fontsize=10)
        self.ax_fitness.set_ylabel('Fitness', color='white', fontsize=10)
        self.ax_fitness.set_title('📈 Fitness Evolution', color='white', fontsize=12, fontweight='bold')
        self.ax_fitness.legend(facecolor='#0a0a0a', edgecolor='#00ffff', labelcolor='white', fontsize=8)
        self.ax_fitness.grid(True, alpha=0.2, color='#333333')
        self.ax_fitness.tick_params(colors='white', labelsize=8)
        for spine in self.ax_fitness.spines.values():
            spine.set_edgecolor('#00ffff')
        
        # 進捗グラフ更新
        self.ax_progress.clear()
        self.ax_progress.set_facecolor('#0a0a0a')
        self.ax_progress.plot(gens, stats_history['progress'], color='#00ffff', linewidth=2, marker='s', markersize=2)
        self.ax_progress.axhline(y=100, color='#ff0000', linestyle='--', linewidth=1.5, alpha=0.7, label='100%')
        self.ax_progress.set_xlabel('Generation', color='white', fontsize=10)
        self.ax_progress.set_ylabel('Progress (%)', color='white', fontsize=10)
        self.ax_progress.set_title('🎯 Lap Progress', color='white', fontsize=12, fontweight='bold')
        self.ax_progress.legend(facecolor='#0a0a0a', edgecolor='#00ffff', labelcolor='white', fontsize=8)
        self.ax_progress.grid(True, alpha=0.2, color='#333333')
        self.ax_progress.tick_params(colors='white', labelsize=8)
        # Y軸の範囲を固定（0-110%）で、進捗の変化を見やすく
        self.ax_progress.set_ylim(0, 110)
        for spine in self.ax_progress.spines.values():
            spine.set_edgecolor('#00ffff')
        
        # 軌跡更新
        self.ax_trajectory.clear()
        self.ax_trajectory.set_facecolor('#0a0a0a')
        self.ax_trajectory.set_aspect('equal')
        self.ax_trajectory.set_xticks([])
        self.ax_trajectory.set_yticks([])
        self.ax_trajectory.set_title('🏁 Best Trajectory', color='white', fontsize=12, fontweight='bold')
        
        # トラック再描画
        tx, ty = self.track[:, 0], self.track[:, 1]
        self.ax_trajectory.plot(tx, ty, color='#333333', linewidth=1.5, alpha=0.6)
        for i in range(0, len(self.track), 15):
            c = plt.Circle(self.track[i], self.half_width, color='#1a1a1a', alpha=0.15, fill=True)
            self.ax_trajectory.add_artist(c)
        
        # ベスト軌跡描画
        trajectory = simulate_for_visualization(best_weights, self.track, self.half_width)
        if len(trajectory) > 1:
            bx, by = zip(*trajectory)
            self.ax_trajectory.plot(bx, by, color='#ffaa00', linewidth=3, alpha=1.0, zorder=10)
            self.ax_trajectory.plot(bx[0], by[0], 'o', color='#00ff00', markersize=12, zorder=11, markeredgecolor='white', markeredgewidth=1.5)
            self.ax_trajectory.plot(bx[-1], by[-1], 'o', color='#ff0000', markersize=12, zorder=11, markeredgecolor='white', markeredgewidth=1.5)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
    
    def close(self):
        """プロットを閉じる"""
        if self.fig:
            plt.close(self.fig)

# ==========================================
# チェックポイント管理
# ==========================================
def save_checkpoint(generation, population, best_individual, stats_history, filename=None):
    """チェックポイントを保存"""
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    if filename is None:
        filename = f"checkpoint_gen_{generation}.pkl"
    
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    checkpoint = {
        'generation': generation,
        'population': population,
        'best_individual': best_individual,
        'stats_history': stats_history,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"💾 Checkpoint saved: {filepath}")
    return filepath

def load_checkpoint(filename=None):
    """チェックポイントを読み込み"""
    if filename is None:
        # 最新のチェックポイントを探す
        if not os.path.exists(CHECKPOINT_DIR):
            return None
        
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pkl')]
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x)))
        filename = checkpoints[-1]
    
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)
    
    print(f"📂 Checkpoint loaded: {filepath}")
    print(f"   Generation: {checkpoint['generation']}")
    print(f"   Saved at: {checkpoint['timestamp']}")
    
    return checkpoint

# ==========================================
# ベスト個体用の詳細評価（ラップ率・ラップタイム）
# ==========================================
def evaluate_lap_metrics(weights, track, half_width):
    """
    ベスト個体用の詳細評価:
    - ラップ完了率（指定方向での累積進捗 0〜複数周可能）
    - ラップタイム（コース内にいた時間[s]）
    """
    x, y = track[0, 0], track[0, 1]
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0

    n_points = len(track)
    max_idx_reached = 0  # 到達した最大インデックス
    max_progress = 0.0
    steps = 0

    for _ in range(SIM_STEPS):
        sensor_readings = sense_jit(x, y, theta, track, half_width,
                                    SENSOR_ANGLES, MAX_SENSOR_DIST)

        nn_input = np.empty(NIN)
        for i in range(NSENS):
            nn_input[i] = sensor_readings[i]
        nn_input[NSENS] = v / MAX_SPEED

        outputs = nn_forward_jit(weights, nn_input, NIN, NHID, NOUT)
        steer = max(-1.0, min(1.0, outputs[0])) * MAX_STEER
        throttle = max(-1.0, min(1.0, outputs[1]))

        v += throttle * THROTTLE_POWER * DT
        v = max(-1.0, min(MAX_SPEED, v))
        theta += (v * math.tan(steer) / WHEELBASE) * DT
        x += v * math.cos(theta) * DT
        y += v * math.sin(theta) * DT

        if distance_to_track_jit(x, y, track) > half_width:
            break

        idx = nearest_track_index_jit(x, y, track)

        # 進捗更新：最大到達インデックスを記録
        forward_dist = (idx - max_idx_reached + n_points) % n_points
        
        if forward_dist > 0 and forward_dist < n_points // 2:
            # 前進している
            if idx > max_idx_reached or (idx < n_points // 4 and max_idx_reached > 3 * n_points // 4):
                # 通常の前進、またはラップ境界を超えた
                current_lap = int(max_progress)  # 現在何周目か
                lap_progress = current_lap + (idx / n_points)
                
                if lap_progress > max_progress:
                    max_progress = lap_progress
                    max_idx_reached = idx

        steps += 1

    lap_time = steps * DT
    return max_progress, lap_time

# ==========================================
# DEAP遺伝的アルゴリズム
# ==========================================
def eval_individual(individual):
    """個体評価関数（マルチプロセス用）"""
    weights = np.array(individual, dtype=np.float64)
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

    # 遺伝子と個体の生成（Xavier初期化風）
    def attr_float_xavier():
        return np.random.normal(0, 0.3)
    
    toolbox.register("attr_float", attr_float_xavier)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=N_WEIGHTS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 遺伝的操作
    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend交叉
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=5)  # より厳しい選択

    return toolbox

def evolve_with_deap(track, half_width, track_laps, resume_from_checkpoint=False, realtime_viz=True):
    """DEAPで進化（改良版：エリート保存、適応的突然変異、チェックポイント、リアルタイム表示）"""
    global TRACK, HALF_WIDTH, TRACK_LAPS
    TRACK = track
    HALF_WIDTH = half_width
    TRACK_LAPS = track_laps

    print(f"Neural Network: {NIN} inputs -> {NHID} hidden -> {NOUT} outputs (ReLU+tanh)")
    print(f"Total weights: {N_WEIGHTS}")
    print(f"Population: {POP_SIZE}, Generations: {GENERATIONS}, Elite: {ELITE_SIZE}")
    print(f"Track points: {len(track)}, Half width: {half_width}")
    print(f"CPU cores: {cpu_count()}")
    print(f"Realtime visualization: {'ON' if realtime_viz else 'OFF'}")
    print()

    # リアルタイム可視化の初期化
    visualizer = None
    if realtime_viz:
        print("🎬 Starting realtime visualization...")
        visualizer = RealtimeVisualizer(track, half_width)

    toolbox = init_deap()

    # マルチプロセスプールの登録
    pool = Pool(processes=cpu_count())
    toolbox.register("map", pool.map)

    # チェックポイントから再開
    start_gen = 0
    stats_history = {'gen': [], 'avg': [], 'max': [], 'min': [], 'progress': []}
    
    if resume_from_checkpoint:
        checkpoint = load_checkpoint()
        if checkpoint:
            pop = checkpoint['population']
            start_gen = checkpoint['generation'] + 1
            stats_history = checkpoint['stats_history']
            print(f"🔄 Resuming from generation {start_gen}\n")
        else:
            print("⚠️  No checkpoint found, starting fresh\n")
            pop = toolbox.population(n=POP_SIZE)
    else:
        pop = toolbox.population(n=POP_SIZE)

    # 統計
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)

    # 殿堂（ベスト保存）
    hof = tools.HallOfFame(ELITE_SIZE)

    # 初期集団を評価（または再開時はスキップ）
    if start_gen == 0:
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)
        record = stats.compile(pop)

        print("gen\tnevals\tavg\t\tmin\t\tmax\t\tcur_prog(%)\tbest_time(s)\tmut_rate")

        # 世代0のベストのラップ情報
        best_ind = hof[0]
        best_w = np.array(best_ind, dtype=np.float64)
        best_prog, best_time = evaluate_lap_metrics(best_w, track, half_width)
        
        # 進捗率を0-100%の範囲に正規化（TRACK_LAPS周完了で100%）
        progress_percent = (best_prog / TRACK_LAPS) * 100
        
        print(f"0\t{len(invalid_ind)}\t"
              f"{record['avg']:.4f}\t{record['min']:.4f}\t{record['max']:.4f}\t"
              f"{progress_percent:.2f}\t\t{best_time:.2f}\t\t{MUTPB_INITIAL:.3f}")
        
        stats_history['gen'].append(0)
        stats_history['avg'].append(record['avg'])
        stats_history['max'].append(record['max'])
        stats_history['min'].append(record['min'])
        stats_history['progress'].append(progress_percent)
        
        # 初期状態を表示
        if visualizer:
            visualizer.update(stats_history, best_w)

    # 進化ループ
    for gen in range(start_gen, GENERATIONS):
        # 適応的変異率（世代に応じて減衰）
        progress = gen / GENERATIONS
        current_mutpb = MUTPB_INITIAL + (MUTPB_FINAL - MUTPB_INITIAL) * progress
        
        # 交叉と突然変異
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=current_mutpb)

        # 無効個体の評価
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # エリート保存戦略：上位ELITE_SIZE個体を必ず次世代に残す
        hof.update(pop)
        elite = list(hof)
        
        # 次世代選択（エリートを除いた数だけ選択）
        offspring_selected = toolbox.select(offspring, len(offspring) - len(elite))
        
        # エリート + 選択された個体
        pop = elite + offspring_selected

        # 殿堂更新
        hof.update(pop)

        # 統計
        record = stats.compile(pop)

        # 現世代ベストのラップ率・ラップタイム
        best_ind = hof[0]
        best_w = np.array(best_ind, dtype=np.float64)
        best_prog, best_time = evaluate_lap_metrics(best_w, track, half_width)

        # 進捗率を0-100%の範囲に正規化（TRACK_LAPS周完了で100%）
        progress_percent = (best_prog / TRACK_LAPS) * 100
        
        print(f"{gen+1}\t{len(invalid_ind)}\t"
              f"{record['avg']:.4f}\t{record['min']:.4f}\t{record['max']:.4f}\t"
              f"{progress_percent:.2f}\t\t{best_time:.2f}\t\t{current_mutpb:.3f}")

        stats_history['gen'].append(gen + 1)
        stats_history['avg'].append(record['avg'])
        stats_history['max'].append(record['max'])
        stats_history['min'].append(record['min'])
        stats_history['progress'].append(progress_percent)

        # リアルタイム表示更新
        if visualizer and (gen + 1) % REALTIME_UPDATE_INTERVAL == 0:
            visualizer.update(stats_history, best_w)

        # チェックポイント保存
        if (gen + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(gen, pop, best_ind, stats_history)

    pool.close()
    pool.join()

    # 最終更新
    if visualizer:
        visualizer.update(stats_history, best_w)
        print("\n🎬 Keeping realtime visualization open...")

    # 最終チェックポイント保存
    save_checkpoint(GENERATIONS - 1, pop, hof[0], stats_history, "checkpoint_final.pkl")

    # 最終ベスト個体
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print("\n✅ Evolution complete!")
    print(f"Best fitness: {best_fitness:.4f} (progress+speed+smooth+safety-penalty)")

    best_weights_arr = np.array(best_individual, dtype=np.float64)
    best_progress, best_lap_time = evaluate_lap_metrics(best_weights_arr, track, half_width)
    laps_completed = int(best_progress)
    remaining_progress = (best_progress - laps_completed) * 100
    print(f"Best lap completion: {laps_completed} lap(s) + {remaining_progress:.1f}% (total progress: {best_progress:.3f})")
    print(f"Best lap time (sim): {best_lap_time:.2f}s")

    return best_weights_arr, pop, stats_history, visualizer

# ==========================================
# 可視化用シミュレーション
# ==========================================
def simulate_for_visualization(weights, track, half_width):
    """可視化用の軌跡生成"""
    x, y = track[0]
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0
    trajectory = [(float(x), float(y))]

    for _ in range(SIM_STEPS):
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

def visualize_results(track, half_width, best_weights, population, stats_history=None):
    """結果可視化（改良版：進捗グラフ追加）"""
    print("\nGenerating visualization...")

    # ベスト軌跡
    best_trajectory = simulate_for_visualization(best_weights, track, half_width)

    # サンプル軌跡
    sample_trajectories = []
    sample_size = min(30, len(population))
    sample_indices = np.random.choice(len(population), sample_size, replace=False)

    for idx in sample_indices:
        weights = np.array(population[idx], dtype=np.float64)
        traj = simulate_for_visualization(weights, track, half_width)
        sample_trajectories.append(traj)

    # 2x2のサブプロット作成
    fig = plt.figure(figsize=(20, 18))
    
    # 1. トラック＋軌跡
    ax1 = plt.subplot(2, 2, (1, 3))  # 左側全体
    ax1.set_facecolor('#000000')
    
    # トラック
    tx, ty = track[:, 0], track[:, 1]
    ax1.plot(tx, ty, color='#333333', linewidth=2, alpha=0.6, label='Track Center')

    for i in range(0, len(track), 10):
        c = plt.Circle(track[i], half_width, color='#1a1a1a', alpha=0.15, fill=True)
        ax1.add_artist(c)

    # サンプル軌跡
    for traj in sample_trajectories:
        if len(traj) > 1:
            px, py = zip(*traj)
            ax1.plot(px, py, color='#00ffff', linewidth=0.5, alpha=0.2)

    # ベスト軌跡
    if len(best_trajectory) > 1:
        bx, by = zip(*best_trajectory)
        ax1.plot(bx, by, color='#ffaa00', linewidth=4.5, alpha=1.0,
                label=f'Best AI ({len(best_trajectory)} steps)', zorder=10)
        ax1.plot(bx[0], by[0], 'o', color='#00ff00', markersize=18,
                label='Start', zorder=11, markeredgecolor='white', markeredgewidth=2.5)
        ax1.plot(bx[-1], by[-1], 'o', color='#ff0000', markersize=18,
                label='End', zorder=11, markeredgecolor='white', markeredgewidth=2.5)

    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.legend(loc='upper right', facecolor='#0a0a0a', edgecolor='#00ffff',
              labelcolor='white', fontsize=11, framealpha=0.95)
    ax1.set_title('🏁 Best AI Trajectory', color='white', fontsize=16, pad=15, fontweight='bold')

    # 2. フィットネス進化グラフ
    if stats_history and len(stats_history['gen']) > 0:
        ax2 = plt.subplot(2, 2, 2)
        ax2.set_facecolor('#0a0a0a')
        
        gens = stats_history['gen']
        ax2.plot(gens, stats_history['max'], color='#00ff00', linewidth=2.5, label='Max Fitness', marker='o', markersize=3)
        ax2.plot(gens, stats_history['avg'], color='#ffaa00', linewidth=2, label='Avg Fitness', alpha=0.8)
        ax2.plot(gens, stats_history['min'], color='#ff4444', linewidth=1.5, label='Min Fitness', alpha=0.6)
        
        ax2.set_xlabel('Generation', color='white', fontsize=11)
        ax2.set_ylabel('Fitness', color='white', fontsize=11)
        ax2.set_title('📈 Fitness Evolution', color='white', fontsize=14, fontweight='bold')
        ax2.legend(facecolor='#0a0a0a', edgecolor='#00ffff', labelcolor='white', fontsize=10)
        ax2.grid(True, alpha=0.2, color='#333333')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#00ffff')
        
        # 3. 進捗率グラフ
        ax3 = plt.subplot(2, 2, 4)
        ax3.set_facecolor('#0a0a0a')
        
        ax3.plot(gens, stats_history['progress'], color='#00ffff', linewidth=2.5, marker='s', markersize=3)
        ax3.axhline(y=100, color='#ff0000', linestyle='--', linewidth=2, alpha=0.7, label='100% Complete')
        
        ax3.set_xlabel('Generation', color='white', fontsize=11)
        ax3.set_ylabel('Progress (%)', color='white', fontsize=11)
        ax3.set_title('🎯 Lap Progress', color='white', fontsize=14, fontweight='bold')
        ax3.legend(facecolor='#0a0a0a', edgecolor='#00ffff', labelcolor='white', fontsize=10)
        ax3.grid(True, alpha=0.2, color='#333333')
        ax3.tick_params(colors='white')
        # Y軸の範囲を固定（0-110%）で、進捗の変化を見やすく
        ax3.set_ylim(0, 110)
        for spine in ax3.spines.values():
            spine.set_edgecolor('#00ffff')

    fig.patch.set_facecolor('#000000')
    
    title = '🚀 Ultra-Fast GA Evolution (DEAP'
    if NUMBA_AVAILABLE:
        title += ' + Numba JIT'
    title += ' + Multiprocessing + Elite + Adaptive Mutation)'

    fig.suptitle(title, color='white', fontsize=18, fontweight='bold', y=0.98)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_improved.png"
    plt.savefig(output_filename, facecolor='#000000', dpi=150, bbox_inches='tight')
    print(f"💾 Saved: {output_filename}")

    plt.show()

# ==========================================
# メイン
# ==========================================
def main():
    print("=" * 75)
    print("  🚀 Ultra-Fast Genetic Algorithm - Neural Network Evolution (IMPROVED)")
    print(f"  Libraries: DEAP + {'Numba JIT + ' if NUMBA_AVAILABLE else ''}Multiprocessing")
    print("  Features: Elite Preservation, Adaptive Mutation, Checkpoints, Realtime Viz")
    print("=" * 75)
    print()

    # コマンドライン引数的な設定（必要に応じて変更）
    TRACK_TYPE = "circuit"  # "circuit", "oval", "figure8"
    RESUME = False  # Trueにするとチェックポイントから再開
    REALTIME_VIZ = True  # リアルタイム表示ON/OFF

    # トラック生成
    TRACK_LAPS_SETTING = 2  # ラップ数設定
    print(f"Generating track (type: {TRACK_TYPE}, laps: {TRACK_LAPS_SETTING})...")
    track, half_width = generate_track(track_type=TRACK_TYPE, laps=TRACK_LAPS_SETTING)
    print(f"Track: {len(track)} points\n")

    # 進化
    start_time = datetime.now()
    best_weights, final_pop, stats_history, visualizer = evolve_with_deap(
        track, half_width, TRACK_LAPS_SETTING,
        resume_from_checkpoint=RESUME,
        realtime_viz=REALTIME_VIZ
    )
    end_time = datetime.now()

    elapsed = (end_time - start_time).total_seconds()
    print(f"\n⏱️  Time: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    if elapsed > 0:
        print(f"⚡ Speed: {GENERATIONS/elapsed:.2f} generations/sec")

    # ベスト個体を保存
    best_filename = f"best_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy"
    np.save(best_filename, best_weights)
    print(f"💾 Best weights saved: {best_filename}")

    # 最終結果の詳細可視化
    print("\n📊 Generating final detailed visualization...")
    visualize_results(track, half_width, best_weights, final_pop, stats_history)
    
    # リアルタイム可視化ウィンドウをクローズ
    if visualizer:
        input("\n👉 Press Enter to close realtime visualization and exit...")
        visualizer.close()

if __name__ == '__main__':
    main()
