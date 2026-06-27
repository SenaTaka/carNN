/*
 * sim.js — carNN シミュレーション + 遺伝的アルゴリズムの JavaScript 移植
 *
 * g_save.py の Numba JIT 関数（simulate_car_jit / sense_jit / nn_forward_jit など）と
 * DEAP の GA 操作をブラウザ用に純粋関数として再実装したもの。
 * DOM には一切触れないため、メインスレッドと Web Worker の両方から利用できる。
 */
(function () {
  'use strict';

  // ====== パラメータ（g_save.py と一致）======
  const NHID = 24;
  const NOUT = 2;
  const SENSOR_ANGLES = [-1.2, -0.7, -0.3, 0.0, 0.3, 0.7, 1.2];
  const NSENS = SENSOR_ANGLES.length; // 7
  const NIN = NSENS + 1;              // 8
  const N_WEIGHTS = NHID * NIN + NHID + NOUT * NHID + NOUT; // 266

  const SIM_STEPS = 30000;
  const DT = 0.001;
  const MAX_SENSOR_DIST = 60.0;
  const WHEELBASE = 2.5;
  const MAX_SPEED = (300 * 1000) / 3600; // m/s
  const MAX_STEER = 0.6;
  const THROTTLE_POWER = 10;

  // ====== ユーティリティ ======
  // Box-Muller 法による正規乱数
  function gaussian(mu, sigma) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // 点(x,y) からトラック折れ線までの最短距離（Python の distance_to_track_jit と同じ間引き）
  function distanceToTrack(x, y, track, n) {
    let minDist = 1e18;
    const step = 4;
    for (let i = 0; i < n; i += step) {
      const i2 = (i + 1) % n;
      const p1x = track[2 * i], p1y = track[2 * i + 1];
      const p2x = track[2 * i2], p2y = track[2 * i2 + 1];
      const vx = p2x - p1x, vy = p2y - p1y;
      const wx = x - p1x, wy = y - p1y;
      const c = vx * vx + vy * vy;
      let d;
      if (c < 1e-10) {
        d = (x - p1x) * (x - p1x) + (y - p1y) * (y - p1y);
      } else {
        let t = (vx * wx + vy * wy) / c;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        const dx = x - (p1x + t * vx);
        const dy = y - (p1y + t * vy);
        d = dx * dx + dy * dy;
      }
      if (d < minDist) minDist = d;
    }
    return Math.sqrt(minDist);
  }

  // 点(x,y) と セグメント i→(i+1) の距離の2乗
  function pointSegDist2(x, y, i, track, n) {
    const i2 = (i + 1) % n;
    const p1x = track[2 * i], p1y = track[2 * i + 1];
    const p2x = track[2 * i2], p2y = track[2 * i2 + 1];
    const vx = p2x - p1x, vy = p2y - p1y;
    const wx = x - p1x, wy = y - p1y;
    const c = vx * vx + vy * vy;
    if (c < 1e-10) return wx * wx + wy * wy;
    let t = (vx * wx + vy * wy) / c;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const dx = x - (p1x + t * vx);
    const dy = y - (p1y + t * vy);
    return dx * dx + dy * dy;
  }

  // 空間グリッドを構築して S に取り付ける（距離クエリの高速化）。
  // セル幅 G = halfWidth + 最大セグメント長 とすることで、
  // halfWidth 以内にあるセグメントは必ず注目セルの 3x3 近傍に登録される。
  function attachGrid(S) {
    const track = S.track, n = S.n, hw = S.halfWidth;
    let maxSeg = 0;
    for (let i = 0; i < n; i++) {
      const i2 = (i + 1) % n;
      const dx = track[2 * i2] - track[2 * i];
      const dy = track[2 * i2 + 1] - track[2 * i + 1];
      const l = Math.sqrt(dx * dx + dy * dy);
      if (l > maxSeg) maxSeg = l;
    }
    const G = Math.max(hw + maxSeg, 1e-3);
    // セル範囲を求める
    let minCx = Infinity, maxCx = -Infinity, minCy = Infinity, maxCy = -Infinity;
    for (let i = 0; i < n; i++) {
      const cx = Math.floor(track[2 * i] / G), cy = Math.floor(track[2 * i + 1] / G);
      if (cx < minCx) minCx = cx;
      if (cx > maxCx) maxCx = cx;
      if (cy < minCy) minCy = cy;
      if (cy > maxCy) maxCy = cy;
    }
    // 近傍参照(3x3)が範囲外アクセスしないよう1セル分の余白を取る
    minCx -= 1; minCy -= 1; maxCx += 1; maxCy += 1;
    const gw = maxCx - minCx + 1;
    const gh = maxCy - minCy + 1;
    const buckets = new Array(gw * gh);
    function add(cx, cy, i) {
      const idx = (cx - minCx) + (cy - minCy) * gw;
      let arr = buckets[idx];
      if (!arr) { arr = []; buckets[idx] = arr; }
      arr.push(i);
    }
    for (let i = 0; i < n; i++) {
      const i2 = (i + 1) % n;
      const cx1 = Math.floor(track[2 * i] / G), cy1 = Math.floor(track[2 * i + 1] / G);
      add(cx1, cy1, i);
      const cx2 = Math.floor(track[2 * i2] / G), cy2 = Math.floor(track[2 * i2 + 1] / G);
      if (cx2 !== cx1 || cy2 !== cy1) add(cx2, cy2, i);
    }
    S.buckets = buckets;
    S.G = G;
    S.gw = gw; S.gh = gh;
    S.minCx = minCx; S.minCy = minCy;
    S.hw2 = hw * hw;
    return S;
  }

  // (x,y) がコース幅(halfWidth)以内にあるか。フラットグリッドで近傍セグメントのみ判定
  function withinTrack(x, y, S) {
    const G = S.G, buckets = S.buckets, track = S.track, n = S.n, hw2 = S.hw2;
    const gw = S.gw, gh = S.gh, minCx = S.minCx, minCy = S.minCy;
    const cx = Math.floor(x / G) - minCx;
    const cy = Math.floor(y / G) - minCy;
    if (cx < 1 || cy < 1 || cx >= gw - 1 || cy >= gh - 1) return false; // 範囲外は確実にコース外
    for (let dy = -1; dy <= 1; dy++) {
      const row = (cy + dy) * gw;
      for (let dx = -1; dx <= 1; dx++) {
        const arr = buckets[cx + dx + row];
        if (!arr) continue;
        for (let k = 0; k < arr.length; k++) {
          if (pointSegDist2(x, y, arr[k], track, n) <= hw2) return true;
        }
      }
    }
    return false;
  }

  // 最近傍のトラックインデックス（last_idx 周辺を探索）
  function getNearestIdx(x, y, track, n, lastIdx) {
    let best = -1, min = 1e18;
    if (lastIdx === -1) {
      for (let i = 0; i < n; i += 5) {
        const dx = x - track[2 * i], dy = y - track[2 * i + 1];
        const d = dx * dx + dy * dy;
        if (d < min) { min = d; best = i; }
      }
    } else {
      for (let r = lastIdx - 100; r < lastIdx + 100; r++) {
        const i = ((r % n) + n) % n;
        const dx = x - track[2 * i], dy = y - track[2 * i + 1];
        const d = dx * dx + dy * dy;
        if (d < min) { min = d; best = i; }
      }
    }
    return best;
  }

  // 線分交差判定（ゴールライン通過用）
  function isIntersect(p1x, p1y, p2x, p2y, q1x, q1y, q2x, q2y) {
    function ccw(ax, ay, bx, by, cx, cy) {
      return (by - ay) * (cx - ax) - (bx - ax) * (cy - ay);
    }
    const d1 = ccw(p1x, p1y, p2x, p2y, q1x, q1y);
    const d2 = ccw(p1x, p1y, p2x, p2y, q2x, q2y);
    const d3 = ccw(q1x, q1y, q2x, q2y, p1x, p1y);
    const d4 = ccw(q1x, q1y, q2x, q2y, p2x, p2y);
    return (
      ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) &&
      ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))
    );
  }

  // 7本のレーザーセンサー。out[0..NSENS-1] に正規化距離を書き込む
  function sense(x, y, theta, S, out) {
    for (let i = 0; i < NSENS; i++) {
      const a = theta + SENSOR_ANGLES[i];
      const ca = Math.cos(a), sa = Math.sin(a);
      let d = 0;
      let hit = false;
      while (d < MAX_SENSOR_DIST) {
        const px = x + d * ca, py = y + d * sa;
        if (!withinTrack(px, py, S)) { hit = true; break; } // 壁の外
        d += 5.0;
      }
      out[i] = (hit ? d : MAX_SENSOR_DIST) / MAX_SENSOR_DIST;
    }
  }

  // 3層NN順伝播（ReLU → tanh）。weights のレイアウトは g_save.py と一致
  function nnForward(w, inputs) {
    const h = new Float64Array(NHID);
    for (let i = 0; i < NHID; i++) {
      let val = 0;
      const base = i * NIN;
      for (let j = 0; j < NIN; j++) val += w[base + j] * inputs[j];
      h[i] = val;
    }
    let idx = NHID * NIN;
    for (let i = 0; i < NHID; i++) {
      const val = h[i] + w[idx + i];
      h[i] = val > 0 ? val : 0; // ReLU
    }
    idx += NHID;
    const out = new Float64Array(NOUT);
    for (let i = 0; i < NOUT; i++) {
      let val = 0;
      const base = idx + i * NHID;
      for (let j = 0; j < NHID; j++) val += w[base + j] * h[j];
      out[i] = val;
    }
    idx += NHID * NOUT;
    for (let i = 0; i < NOUT; i++) out[i] = Math.tanh(out[i] + w[idx + i]);
    return out;
  }

  // 1個体のシミュレーション。record=true なら走行軌跡 [x,y,theta] を返す
  function simulateCar(w, S, record) {
    const track = S.track, n = S.n, halfWidth = S.halfWidth;
    const pointsPerLap = S.pointsPerLap, goal = S.goal;
    let x = track[0], y = track[1];
    let theta = Math.atan2(track[2 * 5 + 1] - y, track[2 * 5] - x);
    let v = 0;
    let lastIdx = 0, totalIdx = 0, finished = false, step = 0;
    const gx1 = goal[0], gy1 = goal[1], gx2 = goal[2], gy2 = goal[3];
    const inputs = new Float64Array(NIN);
    const traj = record ? [] : null;

    for (let s = 0; s < SIM_STEPS; s++) {
      step = s + 1;
      const prevX = x, prevY = y;
      if (record) traj.push([x, y, theta]);

      sense(x, y, theta, S, inputs);
      inputs[NSENS] = v / MAX_SPEED;
      const out = nnForward(w, inputs);
      const steer = out[0] * MAX_STEER;
      const throttle = out[1];

      if (throttle > 0) v += throttle * THROTTLE_POWER * DT;
      else v += throttle * THROTTLE_POWER * 2.0 * DT;
      v = Math.max(-5.0, Math.min(MAX_SPEED, v));
      theta += (v * Math.tan(steer) / WHEELBASE) * DT;
      x += v * Math.cos(theta) * DT;
      y += v * Math.sin(theta) * DT;

      if (!withinTrack(x, y, S)) break; // コースアウト

      const curr = getNearestIdx(x, y, track, n, lastIdx);
      let diff = curr - lastIdx;
      if (diff < -n / 2) diff += n;
      else if (diff > n / 2) diff -= n;
      if (diff < 0) totalIdx += diff * 2; // 逆走ペナルティ
      else totalIdx += diff;
      lastIdx = curr;

      if (totalIdx > pointsPerLap * 0.95) {
        if (isIntersect(prevX, prevY, x, y, gx1, gy1, gx2, gy2)) {
          finished = true;
          if (record) traj.push([x, y, theta]);
          break;
        }
      }
    }

    let fitness;
    if (finished) fitness = 20000.0 + (SIM_STEPS - step) * 10.0;
    else {
      fitness = totalIdx;
      if (step < 20) fitness = 0.0;
    }
    return { fitness, finished, time: step * DT, traj };
  }

  // ウェイポイント補間済みの点列 [{x,y},...] と halfWidth から
  // シミュレーション用の構造体を作る（フラット配列・ゴールライン）
  function buildSim(points, halfWidth) {
    const n = points.length;
    const track = new Float64Array(n * 2);
    for (let i = 0; i < n; i++) {
      track[2 * i] = points[i].x;
      track[2 * i + 1] = points[i].y;
    }
    const p0 = points[0];
    const prev = points[n - 1];
    const next = points[1 % n];
    const dx = next.x - prev.x, dy = next.y - prev.y;
    const len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;
    const goal = [
      p0.x + nx * halfWidth, p0.y + ny * halfWidth,
      p0.x - nx * halfWidth, p0.y - ny * halfWidth,
    ];
    const S = { track, n, halfWidth, pointsPerLap: n, goal };
    return attachGrid(S);
  }

  const CARNN = {
    NHID, NOUT, NSENS, NIN, N_WEIGHTS, SENSOR_ANGLES,
    SIM_STEPS, DT, MAX_SENSOR_DIST, WHEELBASE, MAX_SPEED, MAX_STEER, THROTTLE_POWER,
    gaussian, distanceToTrack, getNearestIdx, isIntersect, sense, nnForward,
    simulateCar, buildSim, attachGrid,
  };

  const glob = typeof self !== 'undefined' ? self : globalThis;
  glob.CARNN = CARNN;
})();
