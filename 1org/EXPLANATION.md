# 遺伝的アルゴリズムによるニューラルネットワーク学習 - 詳細解説

## 📋 目次

1. [プロジェクト概要](#プロジェクト概要)
2. [アルゴリズムの仕組み](#アルゴリズムの仕組み)
3. [実装バージョン比較](#実装バージョン比較)
4. [技術詳細](#技術詳細)
5. [高速化テクニック](#高速化テクニック)
6. [使用方法](#使用方法)
7. [パラメータチューニング](#パラメータチューニング)

---

## 🎯 プロジェクト概要

このプロジェクトは、**遺伝的アルゴリズム（GA）** を使用してニューラルネットワークを進化させ、サーキットを自律走行する車をシミュレートします。

### 主な特徴

- **完全自律制御**: センサー情報のみで判断
- **進化的学習**: 学習データ不要、環境との相互作用で学習
- **物理ベースシミュレーション**: キネマティックバイシクルモデル
- **3つの実装バージョン**: Pure Python / GPU加速 / 超高速版

---

## 🧬 アルゴリズムの仕組み

### 1. 遺伝的アルゴリズム（Genetic Algorithm）

生物の進化を模倣した最適化手法です。

```
世代 0: ランダムな個体群を生成
  ↓
世代 1〜N:
  1. 評価    - 各個体をシミュレーションして適応度を計算
  2. 選択    - 優秀な個体を親として選ぶ
  3. 交叉    - 親の遺伝子を組み合わせて子を作る
  4. 変異    - ランダムに遺伝子を変更
  5. 次世代  - 新しい世代を作成
  ↓
最終世代: 最も優秀な個体が解
```

### 2. ニューラルネットワーク構造

```
入力層 (6ノード)
  ├─ センサー1 (左前方)
  ├─ センサー2 (左)
  ├─ センサー3 (正面)
  ├─ センサー4 (右)
  ├─ センサー5 (右前方)
  └─ 速度
     ↓
隠れ層 (8ノード, tanh活性化)
     ↓
出力層 (2ノード)
  ├─ ステアリング角 (-1.0 〜 1.0)
  └─ スロットル (-1.0 〜 1.0)
```

**重みの総数**: `6×8 + 8 + 8×2 + 2 = 74個`

これらの重みが「遺伝子」として進化します。

### 3. 車両の物理モデル

**キネマティックバイシクルモデル** を使用：

```python
# 速度更新
v += throttle × THROTTLE_POWER × dt
v = clip(v, -1.0, MAX_SPEED)

# 向き更新（ステアリング）
θ += (v × tan(steer) / WHEELBASE) × dt

# 位置更新
x += v × cos(θ) × dt
y += v × sin(θ) × dt
```

### 4. センサーシステム

5つのレイキャストセンサーで壁までの距離を測定：

```
        センサー3 (正面)
         /    |    \
センサー2/     |     \センサー4
       /      |      \
  左前方      車      右前方
      \       |       /
センサー1     車体    センサー5
```

各センサーは正規化された距離（0.0〜1.0）を返します。

### 5. 適応度関数

車の性能を評価する指標：

```python
適応度 = トラック進行率 (0.0 〜 1.0)

ペナルティ要因:
- コースアウト → シミュレーション停止
- 低速・停止 → 進行率が上がらない
- 逆走      → 進行率が減少
```

---

## 📊 実装バージョン比較

### 1. Pure Python版 (`nn_evolve_pure_python.py`)

**特徴:**
- 標準ライブラリのみ使用
- 理解しやすい実装
- 最も遅い

**向いている人:**
- アルゴリズムを学びたい初心者
- コードの詳細を理解したい人

**実行時間:** 約10〜30分（500世代）

```bash
python nn_evolve_pure_python.py
```

---

### 2. GPU加速版 (`nn_evolve_gpu.py`)

**特徴:**
- NumPy/CuPyでベクトル化
- GPU利用可能（CuPy）
- 行列演算による高速化

**使用ライブラリ:**
- `NumPy` - CPU配列演算
- `CuPy` - GPU配列演算（オプション）

**高速化技術:**
```python
# ベクトル化された距離計算
distances = np.sqrt((track[:, 0] - x)**2 + (track[:, 1] - y)**2)
best_idx = np.argmin(distances)

# 行列演算によるNN
hidden = np.tanh(np.dot(inputs, W1) + b1)
output = np.tanh(np.dot(hidden, W2) + b2)
```

**実行時間:** 約5〜15分（500世代）

```bash
# CPU版
python nn_evolve_gpu.py

# GPU版（CuPyインストール後）
pip install cupy-cuda12x  # CUDA 12.x用
python nn_evolve_gpu.py
```

---

### 3. 超高速版 (`nn_evolve_ultra_fast.py`) ⚡

**特徴:**
- DEAP遺伝的アルゴリズムライブラリ
- Numba JITコンパイル
- マルチプロセス並列化
- **最速の実装**

**使用ライブラリ:**
- `DEAP` - 進化計算専用ライブラリ
- `Numba` - JITコンパイラ（Python→機械語）
- `multiprocessing` - CPU並列化

**高速化技術:**

#### A. DEAP（進化計算ライブラリ）

```python
from deap import base, creator, tools, algorithms

# 遺伝的操作が最適化されている
toolbox.register("mate", tools.cxTwoPoint)      # 交叉
toolbox.register("mutate", tools.mutGaussian)   # 変異
toolbox.register("select", tools.selTournament) # 選択

# 進化アルゴリズムの実行
algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=500)
```

#### B. Numba JITコンパイル

Python関数を機械語に変換して高速化：

```python
@jit(nopython=True, cache=True)
def simulate_car_jit(weights, track, ...):
    # この関数全体がC言語レベルの速度で実行される
    for step in range(SIM_STEPS):
        # センサー計算
        # NN順伝播
        # 物理演算
    return fitness
```

**効果:** 10〜100倍高速化

#### C. マルチプロセシング

全CPUコアで並列評価：

```python
from multiprocessing import Pool, cpu_count

# 全コアで並列実行
pool = Pool(processes=cpu_count())
toolbox.register("map", pool.map)

# 200個体を同時評価
fitnesses = toolbox.map(toolbox.evaluate, population)
```

**効果:** コア数に比例した高速化（8コア→8倍速）

**実行時間:** 約1〜5分（500世代、8コア）

```bash
pip install deap numba
python nn_evolve_ultra_fast.py
```

---

## 🔧 技術詳細

### 遺伝的操作の詳細

#### 1. エリート保存（Elitism）

```python
ELITE_SIZE = 20  # 上位20個体を無条件で次世代へ

# 実装
elite_indices = np.argsort(fitnesses)[-ELITE_SIZE:]
new_population = [population[i] for i in elite_indices]
```

**効果:** 最良解の消失を防ぐ

#### 2. トーナメント選択

```python
TOURNAMENT_SIZE = 3

def tournament_selection(population, fitnesses):
    # ランダムに3個体選び、最も優秀な個体を親とする
    candidates = random.sample(range(len(population)), 3)
    best = max(candidates, key=lambda i: fitnesses[i])
    return population[best]
```

**効果:** 多様性を保ちながら優秀な個体を選択

#### 3. 一点交叉（Single-Point Crossover）

```python
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2
```

```
親1: [a1, a2, a3, | a4, a5, a6]
親2: [b1, b2, b3, | b4, b5, b6]
               ↓交叉点
子1: [a1, a2, a3, | b4, b5, b6]
子2: [b1, b2, b3, | a4, a5, a6]
```

#### 4. ガウス変異

```python
def mutate(genome, mutation_rate=0.05):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            genome[i] += random.gauss(0, 0.3)  # 平均0、標準偏差0.3
```

**効果:** 局所解からの脱出

#### 5. 適応的変異率

```python
# 世代が進むにつれて変異率を減衰
current_mutation_rate = MUTATION_RATE × (1.0 - gen / GENERATIONS)

# 初期: 大きな変異で広く探索
# 後期: 小さな変異で細かく調整
```

---

## ⚡ 高速化テクニック

### 1. Numba JITコンパイル

**原理:** PythonコードをLLVMを使って機械語に変換

```python
# 通常のPython（遅い）
def distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# Numba JIT（速い）
@jit(nopython=True)
def distance_jit(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)
```

**注意点:**
- `nopython=True`: Pythonオブジェクト禁止（最速）
- `cache=True`: コンパイル結果をキャッシュ
- NumPy配列のみサポート（リスト不可）

### 2. ベクトル化（NumPy）

```python
# 遅い（ループ）
distances = []
for point in track:
    d = math.sqrt((point[0] - x)**2 + (point[1] - y)**2)
    distances.append(d)
min_dist = min(distances)

# 速い（ベクトル化）
distances = np.sqrt((track[:, 0] - x)**2 + (track[:, 1] - y)**2)
min_dist = np.min(distances)
```

**効果:** 10〜100倍高速化

### 3. マルチプロセス

```python
# シリアル実行（遅い）
fitnesses = [evaluate(ind) for ind in population]

# 並列実行（速い）
with Pool(cpu_count()) as pool:
    fitnesses = pool.map(evaluate, population)
```

**効果:** CPUコア数倍の高速化

### 4. CuPy（GPU）

```python
import cupy as cp

# CPU（NumPy）
distances = np.sqrt((track[:, 0] - x)**2 + (track[:, 1] - y)**2)

# GPU（CuPy）- 同じインターフェース
distances = cp.sqrt((track[:, 0] - x)**2 + (track[:, 1] - y)**2)
```

**効果:** 大規模データで数倍〜数十倍高速化

---

## 💻 使用方法

### 基本的な実行

```bash
# Pure Python版（最も簡単）
python nn_evolve_pure_python.py

# GPU加速版
python nn_evolve_gpu.py

# 超高速版（推奨）
pip install deap numba
python nn_evolve_ultra_fast.py
```

### GPU利用（オプション）

```bash
# CUDA 11.x
pip install cupy-cuda11x

# CUDA 12.x
pip install cupy-cuda12x

# 実行（自動的にGPU検出）
python nn_evolve_gpu.py
```

### 出力ファイル

実行後、以下のファイルが生成されます：

```
YYYYMMDD_HHMMSS.png  - 可視化結果（タイムスタンプ付き）
```

### 実行中の出力例

```
⚡ Numba JIT compiler enabled
======================================================================
  🚀 Ultra-Fast Genetic Algorithm - Neural Network Evolution
  Libraries: DEAP + Numba JIT + Multiprocessing
======================================================================

Neural Network: 6 inputs -> 8 hidden -> 2 outputs
Total weights: 74
Population: 200, Generations: 500
Track points: 400, Half width: 12.0
CPU cores: 8

Gen 0  : Best=0.0523 (5.2%), Avg=0.0234 (2.3%)
Gen 10 : Best=0.2341 (23.4%), Avg=0.1456 (14.6%)
Gen 20 : Best=0.4523 (45.2%), Avg=0.3234 (32.3%)
Gen 50 : Best=0.7234 (72.3%), Avg=0.5678 (56.8%)
Gen 100: Best=0.8567 (85.7%), Avg=0.7234 (72.3%)
...
Gen 499: Best=0.9823 (98.2%), Avg=0.9156 (91.6%)

✅ Evolution complete!
Best fitness: 0.9823 (98.2% Lap)

⏱️  Time: 127.45s (2.12min)
⚡ Speed: 3.92 generations/sec
```

---

## 🎛️ パラメータチューニング

### 遺伝的アルゴリズムのパラメータ

```python
# 基本パラメータ
POP_SIZE = 200         # 個体数（多い→多様性、少ない→高速）
GENERATIONS = 500      # 世代数（多い→良い解、時間かかる）
ELITE_SIZE = 20        # エリート保存数（10〜20%推奨）

# 遺伝的操作
CXPB = 0.7            # 交叉確率（0.6〜0.9推奨）
MUTPB = 0.2           # 変異確率（0.1〜0.3推奨）
TOURNAMENT_SIZE = 3    # トーナメントサイズ（3〜5推奨）
MUTATION_RATE = 0.05   # 各遺伝子の変異率
```

#### 調整ガイド

| 目的 | 調整 |
|------|------|
| より良い解が欲しい | `GENERATIONS`を増やす（500→1000） |
| 多様性を増やしたい | `POP_SIZE`を増やす、`MUTPB`を上げる |
| 収束を速くしたい | `ELITE_SIZE`を増やす、`CXPB`を上げる |
| 局所解を避けたい | `MUTATION_RATE`を上げる |
| 実行時間を短縮 | `POP_SIZE`, `GENERATIONS`を減らす |

### ニューラルネットワークのパラメータ

```python
NSENS = 5       # センサー数（3〜7推奨）
NHID = 8        # 隠れ層ノード数（6〜16推奨）
NOUT = 2        # 出力（固定）
```

**隠れ層ノード数の影響:**
- 少ない（4〜6）: シンプル、学習速い、能力限定
- 適切（8〜12）: バランス良好
- 多い（16〜32）: 高性能、学習遅い、過学習リスク

### シミュレーションパラメータ

```python
SIM_STEPS = 2000         # シミュレーションステップ数
DT = 0.05                # 時間刻み（秒）
MAX_SENSOR_DIST = 40.0   # センサー最大距離
```

### 車両パラメータ

```python
WHEELBASE = 0.5          # ホイールベース（小さい→機敏）
MAX_SPEED = 8.0          # 最高速度
MAX_STEER = 0.7          # 最大ステアリング角（大きい→曲がりやすい）
THROTTLE_POWER = 3.5     # 加速力
```

---

## 🎨 可視化の見方

生成されるPNG画像の解釈：

```
┌─────────────────────────────────────┐
│  トラック（白い線・円）              │
│    - 中心線: トラックの中央         │
│    - 円: トラックの幅を表現         │
│                                     │
│  他の個体の軌跡（シアン、薄い）      │
│    - 失敗した個体                   │
│    - 多様性を示す                   │
│                                     │
│  ベスト個体（太い黄色/オレンジ）     │
│    - 最も優秀な個体                 │
│    - 緑丸: スタート地点             │
│    - 赤丸: 最終到達地点             │
└─────────────────────────────────────┘
```

### 良い結果の判断基準

✅ **良い結果:**
- ベスト軌跡がトラック中央を滑らかに走行
- 最終到達地点がスタートに近い（ほぼ1周）
- 軌跡が安定している

❌ **悪い結果:**
- すぐにコースアウト
- ジグザグした不安定な走行
- 逆走や停止

---

## 🐛 トラブルシューティング

### 1. 実行が遅い

**解決策:**
```bash
# 超高速版を使用
pip install deap numba
python nn_evolve_ultra_fast.py

# パラメータを減らす
POP_SIZE = 100      # 200→100
GENERATIONS = 250   # 500→250
```

### 2. メモリ不足

**解決策:**
```python
# 個体数を減らす
POP_SIZE = 100

# 可視化のサンプル数を減らす
sample_size = min(20, len(population))  # 50→20
```

### 3. Numbaのエラー

```
TypingError: Failed in nopython mode
```

**解決策:**
```python
# nopython=Falseにする（遅くなるが動作）
@jit(nopython=False, cache=True)
```

### 4. マルチプロセスのエラー

**解決策:**
```python
if __name__ == '__main__':  # 必須！
    main()
```

### 5. CuPyのインストール失敗

**解決策:**
```bash
# CUDAバージョンを確認
nvidia-smi

# 正しいバージョンをインストール
pip install cupy-cuda11x  # CUDA 11.x
pip install cupy-cuda12x  # CUDA 12.x

# それでもダメならNumPy版を使用
python nn_evolve_gpu.py  # 自動的にNumPyにフォールバック
```

---

## 📈 性能比較

### 実行時間比較（500世代、8コアCPU）

| バージョン | 実行時間 | 相対速度 |
|-----------|---------|---------|
| Pure Python | 20〜30分 | 1× |
| GPU (NumPy) | 8〜15分 | 2〜3× |
| GPU (CuPy) | 5〜10分 | 3〜5× |
| Ultra-Fast | **2〜5分** | **10〜15×** |

### メモリ使用量

| バージョン | メモリ | 備考 |
|-----------|--------|------|
| Pure Python | 〜200MB | 最小 |
| GPU (NumPy) | 〜500MB | 配列キャッシュ |
| GPU (CuPy) | 〜1GB | GPU VRAM使用 |
| Ultra-Fast | 〜800MB | マルチプロセス |

---

## 🔬 アルゴリズムの改善案

### 1. 適応度関数の改善

```python
# 現在: 単純な進行率
fitness = progress

# 改善案: 複数要因を考慮
fitness = (
    0.7 × progress +           # 進行率
    0.2 × smoothness +         # 滑らかさ
    0.1 × speed_bonus          # 速度ボーナス
)
```

### 2. より複雑なNN構造

```python
# 現在: 単層
NIN → NHID → NOUT

# 改善案: 多層
NIN → NHID1 → NHID2 → NOUT
6   →  16   →   8   →  2
```

### 3. 動的トポロジー進化（NEAT）

ネットワーク構造自体も進化させる：
- ノード数の増減
- 接続の追加・削除

### 4. 共進化

複数種類の車を同時に進化させる：
- 攻撃的な走行スタイル
- 防御的な走行スタイル

---

## 📚 参考文献・リソース

### 遺伝的アルゴリズム
- [DEAP Documentation](https://deap.readthedocs.io/)
- [遺伝的アルゴリズムの基礎](https://ja.wikipedia.org/wiki/遺伝的アルゴリズム)

### ニューラルネットワーク
- [NEAT (NeuroEvolution of Augmenting Topologies)](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- [Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution)

### 高速化技術
- [Numba Documentation](https://numba.pydata.org/)
- [CuPy Documentation](https://cupy.dev/)
- [Python Multiprocessing](https://docs.python.org/ja/3/library/multiprocessing.html)

### 車両モデル
- [Kinematic Bicycle Model](https://www.coursera.org/lecture/intro-self-driving-cars/lesson-2-the-kinematic-bicycle-model-Bi8yE)

---

## 🎓 学習課題

### 初級
1. パラメータを変更して結果の違いを観察
2. 別のトラック形状で実験
3. センサー数を変更してみる

### 中級
4. 適応度関数に新しい要素を追加
5. 隠れ層を2層にしてみる
6. 異なる変異戦略を実装

### 上級
7. NEATアルゴリズムを実装
8. リアルタイム可視化を追加
9. 強化学習（Q-Learning）と比較

---

## ⚖️ ライセンス

このプロジェクトは教育目的で作成されています。
自由に改変・配布可能です。

---

## 👨‍💻 作者

M1プロジェクト

---

## 🤝 貢献

改善案やバグ報告は歓迎します！

---

**最終更新:** 2025年11月25日
