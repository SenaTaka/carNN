/*
 * worker.js — 遺伝的アルゴリズムの学習ループ（バックグラウンドスレッド）
 *
 * g_save.py の DEAP による進化（varAnd / cxBlend / mutGaussian /
 * selTournament / HallOfFame）を再現する。重い計算をここで回すことで
 * メインスレッド（UI・描画）が固まらない。
 */
importScripts('./sim.js');

let stopFlag = false;

self.onmessage = function (e) {
  const m = e.data;
  if (m.type === 'stop') { stopFlag = true; return; }
  if (m.type === 'start') { stopFlag = false; runGA(m); }
};

function runGA(cfg) {
  const C = self.CARNN;
  const S = C.attachGrid({
    track: cfg.track,
    n: cfg.track.length / 2,
    halfWidth: cfg.halfWidth,
    pointsPerLap: cfg.pointsPerLap,
    goal: cfg.goal,
  });
  const N = C.N_WEIGHTS;
  const popSize = cfg.popSize;
  const GEN = cfg.generations;
  const ELITE = 5;
  const rnd = Math.random;

  // --- 個体操作 ---
  function newInd() {
    const a = new Array(N);
    for (let i = 0; i < N; i++) a[i] = C.gaussian(0, 0.5);
    a.fit = undefined;
    return a;
  }
  function cloneFresh(ind) {
    const a = ind.slice();
    a.fit = undefined; // 子は再評価が必要
    return a;
  }
  function cloneKeepFit(ind) {
    const a = ind.slice();
    a.fit = ind.fit;
    return a;
  }
  function evaluate(list) {
    for (const ind of list) {
      if (ind.fit === undefined) ind.fit = C.simulateCar(ind, S, false).fitness;
    }
  }
  // cxBlend (alpha=0.5)
  function mate(a, b) {
    const alpha = 0.5;
    for (let i = 0; i < N; i++) {
      const g = (1 + 2 * alpha) * rnd() - alpha;
      const x = a[i], y = b[i];
      a[i] = (1 - g) * x + g * y;
      b[i] = g * x + (1 - g) * y;
    }
    a.fit = undefined; b.fit = undefined;
  }
  // mutGaussian (mu=0, sigma=0.25, indpb=0.2)
  function mutate(ind) {
    for (let i = 0; i < N; i++) if (rnd() < 0.2) ind[i] += C.gaussian(0, 0.25);
    ind.fit = undefined;
  }
  // DEAP varAnd (cxpb=0.6, mutpb=0.15)
  function varAnd(pop) {
    const off = pop.map(cloneFresh);
    for (let i = 1; i < off.length; i += 2) {
      if (rnd() < 0.6) mate(off[i - 1], off[i]);
    }
    for (let i = 0; i < off.length; i++) {
      if (rnd() < 0.15) mutate(off[i]);
    }
    return off;
  }
  // selTournament (tournsize=4)
  function selTournament(pop, k) {
    const sel = [];
    for (let i = 0; i < k; i++) {
      let best = null;
      for (let t = 0; t < 4; t++) {
        const c = pop[Math.floor(rnd() * pop.length)];
        if (best === null || c.fit > best.fit) best = c;
      }
      sel.push(cloneKeepFit(best));
    }
    return sel;
  }
  // HallOfFame（上位 ELITE 個体を保持）
  let hof = [];
  function updateHof(cands) {
    for (const c of cands) hof.push(cloneKeepFit(c));
    hof.sort((a, b) => b.fit - a.fit);
    hof = hof.slice(0, ELITE);
  }

  // --- 初期集団 ---
  let pop = [];
  for (let i = 0; i < popSize; i++) pop.push(newInd());
  evaluate(pop);
  updateHof(pop);

  // --- 進化ループ ---
  for (let gen = 1; gen <= GEN; gen++) {
    if (stopFlag) break;
    const off = varAnd(pop);
    evaluate(off);
    pop = selTournament(off.concat(hof), popSize);
    updateHof(pop);

    const best = hof[0];
    const r = C.simulateCar(best, S, false);
    const pct = Math.round((r.fitness / S.pointsPerLap) * 100);
    self.postMessage({
      type: 'progress',
      gen, totalGen: GEN,
      fit: r.fitness, time: r.time, finished: r.finished, pct,
      weights: Array.from(best),
    });
  }

  self.postMessage({ type: 'done', weights: Array.from(hof[0]), fit: hof[0].fit });
}
