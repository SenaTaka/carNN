# 🏎️ AI Racing Car Simulation

遺伝的アルゴリズムとニューラルネットワークを使用した自動運転AIレーシングカーシミュレーション

## ✨ NEW: Web UIトラックエディタ

直感的なビジュアルエディタでレーシングコースを簡単に作成できるようになりました！

![Track Editor](https://github.com/user-attachments/assets/9a4ab819-3861-48ba-8ec1-7a70902356b5)

**クイックスタート:**
1. ブラウザで `track_editor.html` を開く
2. クリック＆ドラッグでコースをデザイン
3. JSONファイルでエクスポート
4. シミュレーションで使用

詳細は [TRACK_EDITOR_GUIDE.md](TRACK_EDITOR_GUIDE.md) を参照。

## 📁 プロジェクト構成

```
carNN/
├── g_save.py                    # メインプログラム（保存/ロード機能付き）
├── track_editor.html            # Web UIトラックエディタ（NEW!）
├── track_loader.py              # トラックJSONローダー（NEW!）
├── TRACK_EDITOR_GUIDE.md        # トラックエディタガイド（NEW!）
├── USAGE_EXAMPLE.md             # 使用例とワークフローガイド（NEW!）
├── requirements.txt             # 必要なライブラリのリスト
├── best_weights_*.npy           # 学習済み重みファイル
├── sample_track.json            # サンプルトラック（複雑なコース）
├── example_simple_oval.json     # サンプルトラック（シンプルな楕円）
├── README.md                    # このファイル
└── *.png                        # 実行結果の可視化画像
```

## 🚀 特徴

### 主な機能

- **🎨 Web UIトラックエディタ**: ドラッグ＆ドロップで直感的にコース作成（NEW!）
- **実車スケールシミュレーション**: 実際の車両パラメータを使用（wheelbase: 2.5m, 最高速度: 300km/h）
- **複雑なサーキット**: Catmull-Rom スプライン補間による滑らかで複雑なコース
- **保存/ロード機能**: 学習済みモデルの保存と読み込みが可能
- **リアルタイム可視化**: 学習進捗とベスト走行のリアルタイム表示
- **高速計算**: Numba JIT コンパイルによる最適化

### 技術スタック

- **遺伝的アルゴリズム (GA)**: DEAP ライブラリによる進化計算
- **ニューラルネットワーク**: 3層（入力層、隠れ層、出力層）
- **並列計算**: マルチプロセッシングによる高速化
- **JIT最適化**: Numba による計算の高速化

## 🎯 使い方

### 必要なライブラリ

**方法1: requirements.txtを使用（推奨）**

```bash
pip install -r requirements.txt
```

**方法2: 個別にインストール**

```bash
pip install numpy matplotlib deap
# numbaは高速化のために推奨（オプション）
pip install numba
```

### 基本的な実行方法

#### 1. 新規学習を開始

```python
# g_save.py の LOAD_MODE を None に設定
LOAD_MODE = None
```

```bash
python3 g_save.py
```

#### 2. 最新の重みファイルから再開

```python
# g_save.py の LOAD_MODE を "latest" に設定
LOAD_MODE = "latest"
```

```bash
python3 g_save.py
```

#### 3. 特定の重みファイルから再開

```python
# g_save.py の LOAD_MODE に ファイル名を指定
LOAD_MODE = "best_weights_20251125_224745.npy"
```

```bash
python3 g_save.py
```

### 重みファイルの管理

プログラムは実行終了時（正常終了、中断いずれも）に自動的に以下の形式で重みファイルを保存します：

```
best_weights_YYYYMMDD_HHMMSS.npy
```

例: `best_weights_20251125_224745.npy`

## ⚙️ パラメータ設定

### 主要パラメータ（g_save.py 内）

```python
# GA設定
POP_SIZE = 150          # 集団サイズ
GENERATIONS = 300       # 世代数
ELITE_SIZE = 15         # エリート個体数

# ニューラルネットワーク
NSENS = 9              # センサー数（実際は7本使用）
NHID = 24              # 隠れ層ニューロン数
NOUT = 2               # 出力数（ステアリング、スロットル）

# シミュレーション
SIM_STEPS = 30000      # 最大ステップ数
DT = 0.001             # タイムステップ（秒）
MAX_SENSOR_DIST = 60.0 # センサー最大距離（m）

# 車両パラメータ
WHEELBASE = 2.5        # ホイールベース（m）
MAX_SPEED = 83.3       # 最高速度（m/s = 300km/h）
MAX_STEER = 0.6        # 最大ステアリング角（rad）
THROTTLE_POWER = 10    # 加速度（m/s²）
```

## 🎮 操作方法

### 実行中

- プログラムは自動的に進化を実行
- リアルタイムでグラフとコース上の走行軌跡を表示
- `Ctrl+C` で中断可能（重みファイルは保存されます）

### 出力情報

コンソールに以下の情報が表示されます：

```
Gen   | Best Fit   | Time(s)    | Status
------------------------------------------------
1     | 1234.5     | 12.34      | Running (45%)
2     | 2345.6     | 11.23      | Running (67%)
...
50    | 25000.0    | 8.45       | 🏁 FINISHED
```

- **Gen**: 現在の世代
- **Best Fit**: 最高適応度（完走時は20000以上）
- **Time(s)**: ラップタイム（秒）
- **Status**: 状態（Running: 走行中、FINISHED: 完走）

## 📊 可視化

プログラムは2つのグラフをリアルタイムで更新します：

1. **左側**: 適応度とラップタイムの推移グラフ
2. **右側**: コース全体とベスト個体の走行軌跡

## 🧠 AI の仕組み

### 入力（センサー）

- 7本のレーザーセンサーで壁までの距離を検出
  - 前方中央、左右30°、左右70°、左右120°
- 現在の速度

### 出力（制御）

- **ステアリング**: -0.6 ～ +0.6 rad
- **スロットル**: -1.0 ～ +1.0（加速/減速）

### 評価関数

車両は以下の基準で評価されます：

1. **進行距離**: トラック上をどれだけ進んだか
2. **完走判定**: ゴールラインを正しく通過したか
3. **完走時間**: 完走した場合、時間が早いほど高評価

### 進化プロセス

1. **初期集団**: ランダムな重みで開始（または既存の重みをロード）
2. **評価**: 各個体をシミュレーションで評価
3. **選択**: トーナメント選択で優秀な個体を選出
4. **交叉**: 個体間で重みを交換
5. **突然変異**: ランダムに重みを変更
6. **エリート保存**: 最高の個体を次世代に確実に残す

## 🔧 カスタマイズ

### 🎨 コース形状の変更（NEW! Web UIエディタ）

**簡単な方法**: Web UIトラックエディタを使用 ✨

1. ブラウザで `track_editor.html` を開く
2. ビジュアルにコースをデザイン（クリック＆ドラッグ）
3. JSONファイルとしてエクスポート
4. `g_save.py` で設定：

```python
# トラック設定
TRACK_JSON = "track.json"  # あなたのカスタムトラック
# または付属のサンプルトラックを使用
# TRACK_JSON = "sample_track.json"          # 複雑なコース
# TRACK_JSON = "example_simple_oval.json"   # シンプルな楕円コース
```

詳細は [TRACK_EDITOR_GUIDE.md](TRACK_EDITOR_GUIDE.md) と [USAGE_EXAMPLE.md](USAGE_EXAMPLE.md) を参照してください。

**従来の方法**: Pythonコードを直接編集

`generate_complex_track()` 関数内の `base_waypoints` を編集：

```python
base_waypoints = np.array([
    [0, -20], [-30, -20], [-20, 20], [-10, 30], [10, 40],
    [80, 80], [100, 70], [90, 50], [110, 30], [90, -10],
    [100, -50], [60, -70], [20, -60], [0, -40],
])
```

### ニューラルネットワーク構造の変更

```python
NHID = 24  # 隠れ層のニューロン数を変更
```

**注意**: 構造を変更した場合、既存の重みファイルは使用できなくなります。

### センサー配置の変更

```python
SENSOR_ANGLES = np.array([-1.2, -0.7, -0.3, 0.0, 0.3, 0.7, 1.2])
```

## 📈 学習のコツ

1. **段階的学習**:
   - まず短い世代数（50-100世代）で学習
   - 最新の重みをロードして追加学習を繰り返す

2. **パラメータ調整**:
   - 完走できない場合: `SIM_STEPS` を増やす
   - 学習が遅い場合: `POP_SIZE` を増やす
   - 過学習の場合: 突然変異率を上げる

3. **継続学習**:
   - `LOAD_MODE = "latest"` で最新の重みから再開
   - 半数の個体を既存モデルの変異版にすることで、学習済みの知識を活用しながら探索を継続

## ⚠️ トラブルシューティング

### Q: "Weights size mismatch" エラーが出る

**A**: ニューラルネットワークの構造（`NSENS`, `NHID`, `NOUT`）が変更されている可能性があります。`LOAD_MODE = None` で新規学習を開始してください。

### Q: プログラムが遅い

**A**: 以下を確認してください：
- Numba が正しくインストールされているか
- マルチプロセスが動作しているか（CPU コア数を確認）

### Q: 完走できない

**A**: 以下を試してください：
- `SIM_STEPS` を増やす（例: 50000）
- より多くの世代数で学習（例: 500世代以上）
- 既存の良い重みファイルから再開

## 📝 ライセンス

このプロジェクトは教育目的で作成されています。

## 🙏 謝辞

- DEAP (Distributed Evolutionary Algorithms in Python)
- Numba (JIT compiler for Python)
- Matplotlib (Visualization library)

---

**Last Updated**: 2025-12-05
