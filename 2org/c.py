#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズムによる自律走行車学習システム (完全版)
特徴:
- 時計回り (Clockwise) コース生成
- 正確なラップ数 (Laps) カウント
- コース境界線（壁）の描画
- Numba + Multiprocessing による高速化
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

# 警告抑制とmatplotlib設定
warnings.filterwarnings('ignore')
plt.ion()

# --- Numba JIT設定 ---
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
    print("⚡ Numba JIT compiler enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("⚠️  Numba not available. Install via 'pip install numba' for speed.")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ==========================================
# パラメータ設定
# ==========================================
# GA設定
POP_SIZE = 200
GENERATIONS = 300
ELITE_SIZE = 15
CXPB = 0.6
MUTPB_INITIAL = 0.4
MUTPB_FINAL = 0.1

# NN構造
NSENS = 7  # センサー数
NIN = NSENS + 1
NHID = 16
NOUT = 2
N_WEIGHTS = NHID * NIN + NHID + NOUT * NHID + NOUT

# シミュレーション設定
SIM_STEPS = 2500       # 制限時間（ステップ数）
DT = 0.05
MAX_SENSOR_DIST = 50.0
# センサー角度（正面0度、左右対称）
SENSOR_ANGLES = np.array([-1.2, -0.7, -0.3, 0.0, 0.3, 0.7, 1.2])

# 車両物理
WHEELBASE = 0.5
MAX_SPEED = 12.0
MAX_STEER = 0.6
THROTTLE_POWER = 5.0

# 表示設定
REALTIME_UPDATE_INTERVAL = 5

# グローバル変数（マルチプロセス共有用）
TRACK = None
HALF_WIDTH = None
POINTS_PER_LAP = None

# ==========================================
# 関数定義: コース生成 & 計算
# ==========================================
def generate_track(track_type="circuit", n_points_per_lap=200, total_laps=3):
    """
    時計回り(Clockwise)のコース座標を生成
    """
    # 時計回りにするために t をマイナス方向へ進める
    t = np.linspace(0, -2 * np.pi * total_laps, n_points_per_lap * total_laps)
    
    if track_type == "circuit":
        # 複雑なサーキット形状
        r = 80.0 + 30.0 * np.sin(3.0 * t) + 15.0 * np.cos(7.0 * t)
        x = r * np.cos(t)
        y = r * np.sin(t)
        width = 16.0
    
    elif track_type == "oval":
        a, b = 120.0, 60.0
        x = a * np.cos(t) + 20.0 * np.sin(5.0 * t)
        y = b * np.sin(t) + 10.0 * np.cos(5.0 * t)
        width = 18.0
        
    else: # default
        x = 100 * np.cos(t)
        y = 100 * np.sin(t)
        width = 15.0
        
    return np.column_stack([x, y]), width, n_points_per_lap

def calculate_track_borders(track, half_width):
    """
    コースの左右の壁（境界線）の座標を計算
    """
    # ループのつなぎ目を滑らかにするため、データを一時的に拡張して勾配計算
    pad = 2
    track_padded = np.vstack([track[-pad:], track, track[:pad]])
    
    dx = np.gradient(track_padded[:, 0])
    dy = np.gradient(track_padded[:, 1])
    
    # 法線ベクトル（進行方向に対して垂直）
    # (dx, dy) -> (-dy, dx) で90度回転
    normals = np.column_stack((-dy, dx))
    
    # 正規化
    norm_lengths = np.linalg.norm(normals, axis=1)
    norm_lengths[norm_lengths == 0] = 1.0
    normals = normals / norm_lengths[:, np.newaxis]
    
    # パディングを除去
    normals = normals[pad:-pad]
    
    # 中心線から左右に展開
    border_inner = track + normals * half_width
    border_outer = track - normals * half_width
    
    return border_inner, border_outer

# ==========================================
# 関数定義: Numba高速化ロジック
# ==========================================
@jit(nopython=True, cache=True)
def get_dist_sq(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2

@jit(nopython=True, cache=True)
def point_to_segment_dist_jit(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c = vx*vx + vy*vy
    if c < 1e-10:
        return math.sqrt(get_dist_sq(px, py, ax, ay))
    t = (vx*wx + vy*wy) / c
    t = max(0.0, min(1.0, t))
    closest_x = ax + t*vx
    closest_y = ay + t*vy
    return math.sqrt(get_dist_sq(px, py, closest_x, closest_y))

@jit(nopython=True, cache=True)
def distance_to_track_jit(x, y, track):
    """コース中心線までの最短距離を計算"""
    min_dist = 1e9
    # 全探索（Numbaなら十分高速）
    for i in range(len(track) - 1):
        d = point_to_segment_dist_jit(x, y, track[i,0], track[i,1], track[i+1,0], track[i+1,1])
        if d < min_dist:
            min_dist = d
    return min_dist

@jit(nopython=True, cache=True)
def get_nearest_idx_jit(x, y, track, last_idx):
    """現在位置に最も近いトラック上の点（インデックス）を探す"""
    n_points = len(track)
    best_idx = -1
    min_dist = 1e18
    
    # 高速化: 前回位置周辺を優先探索
    start_search = 0
    end_search = n_points
    
    # 前回インデックスが有効なら、その前後50ポイントを重点探索
    search_indices = np.arange(n_points) # デフォルト全探索
    if last_idx != -1:
        # -20 ~ +80 の範囲を見る
        indices = np.arange(last_idx - 20, last_idx + 80)
        search_indices = indices % n_points

    for i in search_indices:
        d = (x - track[i, 0])**2 + (y - track[i, 1])**2
        if d < min_dist:
            min_dist = d
            best_idx = i
            
    return best_idx

@jit(nopython=True, cache=True)
def sense_jit(x, y, theta, track, half_width, angles, max_dist):
    """レイキャストセンサー"""
    readings = np.zeros(len(angles))
    for i in range(len(angles)):
        angle = theta + angles[i]
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        
        step = 3.0 # レイを飛ばす刻み幅（粗くして高速化）
        d = 0.0
        hit = False
        
        while d < max_dist:
            px = x + d * cos_a
            py = y + d * sin_a
            dist_center = distance_to_track_jit(px, py, track)
            
            if dist_center > half_width:
                hit = True
                break
            d += step
            
        # 精密化（壁付近で二分探索的調整を行っても良いが、今回は簡易版）
        readings[i] = (d if hit else max_dist) / max_dist
    return readings

@jit(nopython=True, cache=True)
def nn_forward_jit(weights, inputs, nin, nhid, nout):
    """ニューラルネットワーク順伝播 (ReLU + Tanh)"""
    # 重み展開
    idx = 0
    # Layer 1
    w1 = weights[idx : idx + nin*nhid].reshape((nhid, nin))
    idx += nin*nhid
    b1 = weights[idx : idx + nhid]
    idx += nhid
    # Layer 2
    w2 = weights[idx : idx + nhid*nout].reshape((nout, nhid))
    idx += nhid*nout
    b2 = weights[idx : idx + nout]
    
    # Hidden
    h = np.zeros(nhid)
    for i in range(nhid):
        val = b1[i]
        for j in range(nin):
            val += w1[i, j] * inputs[j]
        h[i] = max(0.0, val) # ReLU
        
    # Output
    out = np.zeros(nout)
    for i in range(nout):
        val = b2[i]
        for j in range(nhid):
            val += w2[i, j] * h[j]
        out[i] = math.tanh(val) # -1.0 ~ 1.0
        
    return out

@jit(nopython=True, cache=True)
def simulate_car_jit(weights, track, half_width, points_per_lap, sensor_angles, 
                     sim_steps, dt, max_speed, max_steer, throttle_pwr, wheelbase):
    """
    車両運動シミュレーション & 評価
    """
    # 初期位置（トラックの始点）
    x, y = track[0, 0], track[0, 1]
    # 初期向き（トラック1点目方向＝時計回り）
    theta = math.atan2(track[1, 1] - y, track[1, 0] - x)
    v = 0.0
    
    n_points = len(track)
    last_idx = 0
    
    total_idx_progress = 0  # 累積進捗（インデックス単位）
    max_idx_progress = 0    # 最大到達点
    
    steps_alive = 0
    total_v = 0.0
    
    # 逆走検知
    wrong_way_counter = 0

    for _ in range(sim_steps):
        # 1. センサー取得
        sensors = sense_jit(x, y, theta, track, half_width, sensor_angles, MAX_SENSOR_DIST)
        
        # 2. NN入力作成
        inputs = np.zeros(len(sensors) + 1)
        for i in range(len(sensors)):
            inputs[i] = sensors[i]
        inputs[len(sensors)] = v / max_speed
        
        # 3. NN制御
        outputs = nn_forward_jit(weights, inputs, len(inputs), NHID, NOUT)
        steer = outputs[0] * max_steer
        throttle = outputs[1]
        
        # 4. 物理更新
        if throttle > 0:
            v += throttle * throttle_pwr * dt
        else:
            v += throttle * throttle_pwr * 2.0 * dt # ブレーキ強化
        v = max(-2.0, min(max_speed, v))
        
        theta += (v * math.tan(steer) / wheelbase) * dt
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        
        # 5. 衝突判定
        dist = distance_to_track_jit(x, y, track)
        if dist > half_width:
            break # コースアウト
            
        steps_alive += 1
        total_v += v
        
        # 6. 進捗計算（ラップ対応）
        current_idx = get_nearest_idx_jit(x, y, track, last_idx)
        diff = current_idx - last_idx
        
        # ラップ境界の補正 (例: 199 -> 2 = +3ステップ)
        if diff < -n_points / 2:
            diff += n_points
        elif diff > n_points / 2:
            diff -= n_points
            
        if diff > 0:
            total_idx_progress += diff
            wrong_way_counter = 0
        elif diff < 0:
            total_idx_progress += diff # 後退ペナルティ
            wrong_way_counter += 1
            
        if total_idx_progress > max_idx_progress:
            max_idx_progress = total_idx_progress
            
        last_idx = current_idx
        
        # 逆走しすぎたら終了
        if wrong_way_counter > 60:
            break

    # 評価値計算
    # ラップ数換算
    laps_completed = max_idx_progress / points_per_lap
    avg_speed = total_v / steps_alive if steps_alive > 0 else 0.0
    
    # フィットネス = (周回数 * 100) + (平均速度 * 2) - (早期死亡ペナルティ)
    fitness = laps_completed * 100.0 + avg_speed * 1.5
    
    if steps_alive < 30:
        fitness = 0.0
        
    return fitness, laps_completed

# ==========================================
# 可視化クラス
# ==========================================
class RealtimeVisualizer:
    def __init__(self, track, half_width):
        self.track = track
        self.half_width = half_width
        
        # 背景色設定
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 7))
        self.fig.patch.set_facecolor('#121212')
        
        # 左：グラフ
        self.ax_graph = self.axes[0]
        self.ax_graph.set_facecolor('#1e1e1e')
        self.ax_graph.set_xlabel('Generation')
        self.ax_graph.set_title('Learning Progress')
        self.ax_graph.grid(True, alpha=0.2)
        
        # 右：コース
        self.ax_course = self.axes[1]
        self.ax_course.set_facecolor('#1e1e1e')
        self.ax_course.set_title('Best Trajectory (Clockwise)')
        self.ax_course.set_aspect('equal')
        self.ax_course.axis('off') # 軸を消す
        
        # コース境界線の計算
        self.border_in, self.border_out = calculate_track_borders(track, half_width)
        
        plt.tight_layout()
        plt.pause(0.1)
        
    def update(self, history, best_weights, points_per_lap):
        # --- グラフ更新 ---
        self.ax_graph.clear()
        self.ax_graph.set_facecolor('#1e1e1e')
        self.ax_graph.grid(True, alpha=0.2)
        self.ax_graph.set_xlabel('Generation')
        
        gens = [h['gen'] for h in history]
        fits = [h['fitness'] for h in history]
        laps = [h['laps'] for h in history]
        
        # Fitness (左軸)
        l1 = self.ax_graph.plot(gens, fits, color='#00ff00', label='Fitness')
        self.ax_graph.set_ylabel('Fitness', color='#00ff00')
        self.ax_graph.tick_params(axis='y', colors='#00ff00')
        
        # Laps (右軸)
        ax2 = self.ax_graph.twinx()
        l2 = ax2.plot(gens, laps, color='#00ffff', label='Laps')
        ax2.set_ylabel('Laps Completed', color='#00ffff')
        ax2.tick_params(axis='y', colors='#00ffff')
        
        # --- コース更新 ---
        self.ax_course.clear()
        self.ax_course.set_facecolor('#1e1e1e')
        self.ax_course.set_aspect('equal')
        self.ax_course.axis('off')
        
        # 壁の描画 (白の実線)
        self.ax_course.plot(self.border_in[:,0], self.border_in[:,1], color='white', linewidth=1.5)
        self.ax_course.plot(self.border_out[:,0], self.border_out[:,1], color='white', linewidth=1.5)
        
        # 中心線 (薄い点線)
        self.ax_course.plot(self.track[:,0], self.track[:,1], color='gray', linestyle='--', alpha=0.5)
        
        # スタート地点 (緑の丸)
        self.ax_course.plot(self.track[0,0], self.track[0,1], 'o', color='#00ff00', markersize=8)
        
        # ベスト個体の走行軌跡
        traj = self.simulate_trajectory(best_weights, points_per_lap)
        if len(traj) > 0:
            tx, ty = zip(*traj)
            # 軌跡 (オレンジ〜黄色)
            self.ax_course.plot(tx, ty, color='#ffcc00', linewidth=2.5, alpha=0.9)
            # 終了地点 (赤バツ)
            self.ax_course.plot(tx[-1], ty[-1], 'x', color='#ff0000', markersize=10)
            
        # タイトルに現状表示
        last_lap = laps[-1] if laps else 0
        self.ax_course.set_title(f"Gen {gens[-1]}: {last_lap:.2f} Laps / Best Fit: {fits[-1]:.0f}", color='white')
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        
    def simulate_trajectory(self, weights, points_per_lap):
        """可視化用に1回シミュレーションして軌跡リストを返す"""
        # JIT関数はクラス内から直接呼べないので、グローバル定数等を使用
        # ここはJITなしでも良いが、ロジック共通化のためシミュレーション関数を少し変形して使う手もある
        # 簡易的に、既存のJIT関数と同様の動きをするPythonコードを書くか、
        # あるいは描画専用に少し簡略化したシミュレータを回す
        
        x, y = self.track[0,0], self.track[0,1]
        theta = math.atan2(self.track[1,1]-y, self.track[1,0]-x)
        v = 0.0
        trajectory = []
        
        # JIT関数を呼ぶ（ラッパー）
        # ステップごとの座標を記録したいが、JIT関数はfitnessしか返さないため
        # ここで再度Pythonループで回す（可視化用なので低速でもOK）
        for _ in range(SIM_STEPS):
            trajectory.append((x, y))
            
            readings = sense_jit(x, y, theta, self.track, self.half_width, SENSOR_ANGLES, MAX_SENSOR_DIST)
            
            in_data = np.zeros(len(readings)+1)
            in_data[:len(readings)] = readings
            in_data[-1] = v / MAX_SPEED
            
            out = nn_forward_jit(weights, in_data, len(in_data), NHID, NOUT)
            steer = out[0] * MAX_STEER
            throttle = out[1]
            
            if throttle > 0: v += throttle * THROTTLE_POWER * DT
            else: v += throttle * THROTTLE_POWER * 2.0 * DT
            v = max(-2.0, min(MAX_SPEED, v))
            
            theta += (v * math.tan(steer) / WHEELBASE) * DT
            x += v * math.cos(theta) * DT
            y += v * math.sin(theta) * DT
            
            dist = distance_to_track_jit(x, y, self.track)
            if dist > self.half_width:
                break
                
        return trajectory

# ==========================================
# メイン処理
# ==========================================
def eval_wrapper(individual):
    """マルチプロセス用ラッパー"""
    w = np.array(individual, dtype=np.float64)
    fit, laps = simulate_car_jit(
        w, TRACK, HALF_WIDTH, POINTS_PER_LAP, SENSOR_ANGLES,
        SIM_STEPS, DT, MAX_SPEED, MAX_STEER, THROTTLE_POWER, WHEELBASE
    )
    return fit, # DEAPはタプルを要求

def main():
    global TRACK, HALF_WIDTH, POINTS_PER_LAP
    
    print("\n🏎️  AI RACING EVOLUTION (Clockwise + Walls) 🏎️")
    print("==================================================")
    
    # 1. コース生成
    points_lap = 250
    TRACK, HALF_WIDTH, POINTS_PER_LAP = generate_track("circuit", n_points_per_lap=points_lap, total_laps=3)
    print(f"✅ Track Generated: {len(TRACK)} points ({points_lap} pts/lap)")
    
    # 2. GA設定
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
    
    # マルチプロセス
    pool = Pool(cpu_count())
    toolbox.register("map", pool.map)
    
    # 3. 初期化
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(ELITE_SIZE)
    
    # 可視化準備
    viz = RealtimeVisualizer(TRACK, HALF_WIDTH)
    history = []
    
    print(f"\n{'Gen':<4} | {'Best Fit':<10} | {'Avg Fit':<10} | {'LAPS':<10}")
    print("-" * 45)
    
    # 4. 進化ループ
    for gen in range(1, GENERATIONS + 1):
        # 変異率の減衰
        progress = gen / GENERATIONS
        mutpb = MUTPB_INITIAL + (MUTPB_FINAL - MUTPB_INITIAL) * progress
        
        # 次世代選択・生成
        if gen == 1:
            offspring = pop
        else:
            offspring = toolbox.select(pop, len(pop) - ELITE_SIZE)
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=CXPB, mutpb=mutpb)
            # エリートを戻す
            offspring.extend(hof)
            
        # 評価
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
            
        pop = offspring
        hof.update(pop)
        
        # 統計取得
        best_ind = hof[0]
        fit_vals = [ind.fitness.values[0] for ind in pop]
        max_fit = np.max(fit_vals)
        avg_fit = np.mean(fit_vals)
        
        # ベスト個体のラップ数を再計算（JIT関数から直接取得）
        _, best_laps = simulate_car_jit(
            np.array(best_ind), TRACK, HALF_WIDTH, POINTS_PER_LAP, SENSOR_ANGLES,
            SIM_STEPS, DT, MAX_SPEED, MAX_STEER, THROTTLE_POWER, WHEELBASE
        )
        
        # ログ保存
        history.append({'gen': gen, 'fitness': max_fit, 'laps': best_laps})
        
        print(f"{gen:<4} | {max_fit:<10.1f} | {avg_fit:<10.1f} | {best_laps:<10.2f} laps")
        
        # 描画更新
        if gen % REALTIME_UPDATE_INTERVAL == 0:
            viz.update(history, np.array(best_ind), POINTS_PER_LAP)
            
        # 目標達成判定
        if best_laps > 2.8: # 3周近くしたら終了
            print("\n🎉 GOAL REACHED! The AI mastered the track.")
            break
            
    pool.close()
    pool.join()
    
    # 最終結果
    viz.update(history, np.array(hof[0]), POINTS_PER_LAP)
    np.save("best_racer_final.npy", np.array(hof[0]))
    print("\n💾 Model saved. Press Enter to exit.")
    input()

if __name__ == "__main__":
    main()