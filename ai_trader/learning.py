"""AI Learning Engine — inti 'belajar dari kesalahan'.

Dua mekanisme belajar:
1. ONLINE (trade-by-trade): setiap trade yang CLOSE dievaluasi.
   - Trade rugi  -> bobot strategi yang dominan memberi sinyal saat ENTRY
                   diturunkan (penghukuman), trade profit -> dinaikkan.
   - Aturan stop-loss / take-profit / threshold masuk ikut di-tune
     berdasarkan statistik P&L terakhir (adaptive risk control).
2. OFFLINE (backtest evolusioner): pada window data historis, AI membuat
   varian portofolio (mutate), backtest semua varian, mempertahankan yang
   paling fit — dan menyimpan pelajaran ke experience_buffer.jsonl.
"""

from __future__ import annotations

import json
import os
import time
import numpy as np

from .strategies.portfolio import (
    Strategy, combined_signal, mutate, save_portfolio, portfolio_gates,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EXPERIENCE_PATH = os.path.join(DATA_DIR, "experience_buffer.jsonl")
BRAIN_PATH = os.path.join(DATA_DIR, "brain.json")

from .gates import sync_brain, ENTRY_THR_MIN, ENTRY_THR_MAX

DEFAULT_BRAIN = {
    "entry_threshold": 0.20,   # nilai AWAL — AI yang menyesuaikan sendiri
                               #   (evolusi + safety-net kesunyian), bukan
                               #   template mati manusia (dulu 0.25)
    "stop_loss_pct": 0.02,     # 2%
    "take_profit_pct": 0.035,  # 3.5%
    "alloc_frac": 0.5,         # fraksi kas per posisi
    "lr": 0.10,                # learning rate penyesuaian bobot
    "total_wins": 0,
    "total_losses": 0,
}


def load_brain() -> dict:
    if os.path.exists(BRAIN_PATH):
        try:
            with open(BRAIN_PATH) as f:
                b = json.load(f)
            return sync_brain({**DEFAULT_BRAIN, **b})
        except Exception:
            pass
    return sync_brain(dict(DEFAULT_BRAIN))


def save_brain(brain: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BRAIN_PATH, "w") as f:
        json.dump(brain, f, indent=2)


def log_experience(entry: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(EXPERIENCE_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ================= 1. BELAJAR ONLINE DARI KESALAHAN =================

def dominant_strategies(strategies: list[Strategy], features_at_entry: dict) -> list[tuple[str, float]]:
    """Perkirakan kontribusi tiap strategi pada sinyal entry dari snapshot fitur."""
    contribs = []
    for s in strategies:
        fn_name = s.signal_fn_name
        sig = _quick_signal(fn_name, features_at_entry, s.params)
        contribs.append((s.name, s.weight * sig))
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs


def _quick_signal(fn_name: str, feat: dict, p: dict) -> float:
    """Hitung ulang sinyal dari snapshot fitur (fallback aman)."""
    g = lambda k, d=0.0: float(feat.get(k, d))
    try:
        if fn_name == "sig_rsi_cross":
            rsi = g("rsi")
            return 1.0 if rsi < p["oversold"] / 50 - 1 else (-1.0 if rsi > p["overbought"] / 50 - 1 else 0.0)
        if fn_name == "sig_macd_momentum":
            return float(np.tanh(g("macd_hist") * p["sensitivity"]))
        if fn_name == "sig_ema_trend":
            return float(np.clip(g("ema_trend") * p["sensitivity"], -1, 1))
        if fn_name == "sig_bollinger_reversion":
            bb = g("bb_pos")
            return 1.0 if bb < -p["band"] else (-1.0 if bb > p["band"] else 0.0)
        if fn_name == "sig_stochastic":
            s = g("stoch_k")
            return 1.0 if s < p["oversold"] / 50 - 1 else (-1.0 if s > p["overbought"] / 50 - 1 else 0.0)
        if fn_name == "sig_candle_reversal":
            bull = g("engulfing_bull") + g("hammer") + g("three_white") + g("marubozu_bull")
            bear = g("engulfing_bear") + g("shooting_star") + g("three_black") + g("marubozu_bear")
            return float(np.clip((bull - bear) * p["strength"], -1, 1))
        if fn_name == "sig_volume_confirmation":
            return float(np.clip(g("obv_slope"), -1, 1)) if g("vol_sma_ratio") > p["spike"] else 0.0
        if fn_name == "sig_vwap_deviation":
            return float(np.clip(-g("vwap_dev") * p["sensitivity"], -1, 1))
    except Exception:
        pass
    return 0.0


def learn_from_trade(trade: dict, strategies: list[Strategy], brain: dict) -> list[str]:
    """Update bobot strategi & risk rules berdasarkan satu trade selesai."""
    pnl_pct = trade.get("pnl_pct")
    if pnl_pct is None:
        return []
    feats = trade.get("entry_features", {})
    lr = brain["lr"]
    lessons = []

    contribs = dominant_strategies(strategies, feats)
    # Arah posisi: kita selalu long di paper trader ini; sinyal entry harus >0
    reward = float(np.clip(pnl_pct / 0.03, -1, 1))  # normalisasi: ±3% => ±1

    by_name = {s.name: s for s in strategies}
    for name, c in contribs[:4]:  # hanya 4 kontributor terbesar
        s = by_name.get(name)
        if s is None or c == 0:
            continue
        agree = float(np.sign(c))  # strategi ini menyarankan beli (+) saat entry
        if agree <= 0:
            continue  # hanya penyalahkan strategi yang searah dengan posisi long
        old_w = s.weight
        # trade rugi -> bobot turun, trade profit -> bobot naik (credit assignment)
        s.weight = max(0.05, min(3.0, old_w * (1 + lr * reward)))
        if s.weight > old_w + 1e-6:
            lessons.append(f"{name}: profit -> bobot naik {old_w:.2f}->{s.weight:.2f}")
        elif s.weight < old_w - 1e-6:
            lessons.append(f"{name}: rugi -> bobot turun {old_w:.2f}->{s.weight:.2f}")
        if pnl_pct > 0:
            s.wins += 1
        else:
            s.losses += 1

    # ---- adaptive risk control dari pengalaman ----
    if pnl_pct < 0:
        brain["total_losses"] += 1
        # kerugian besar? perketat stop loss & naikkan ambang masuk
        if pnl_pct < -brain["stop_loss_pct"] * 1.5:
            brain["stop_loss_pct"] = round(max(0.008, brain["stop_loss_pct"] * 0.9), 4)
            lessons.append(f"stop_loss diperketat -> {brain['stop_loss_pct']:.3%}")
        # batas atas dikendurkan 0.7 -> 0.45 (gates.ENTRY_THR_MAX): kenaikan
        # tetap ada (belajar dari rugi) tapi tak boleh mencekik proses belajar
        brain["entry_threshold"] = min(0.45, brain["entry_threshold"] + 0.02)
        lessons.append(f"entry_threshold dinaikkan -> {brain['entry_threshold']:.2f} (lebih selektif)")
    else:
        brain["total_wins"] += 1
        # PROFIT ADALAH GURU TERBAIK: setelah trade untung, AI BERANI
        # menurunkan ambangnya sendiri (bukan cuma tiap 5 win) supaya
        # frekuensi belajar naik — sesuai filosofi "takut-takut kapan belajar".
        old_thr = brain["entry_threshold"]
        brain["entry_threshold"] = max(0.10, round(old_thr - 0.01, 3))
        if brain["entry_threshold"] < old_thr - 1e-6:
            lessons.append(f"profit -> entry_threshold DITURUNKAN sendiri "
                           f"{old_thr:.2f}->{brain['entry_threshold']:.2f} "
                           f"(perbanyak pengalaman)")
        if brain["total_wins"] % 5 == 0:
            brain["entry_threshold"] = max(0.10, round(brain["entry_threshold"] - 0.01, 3))
            lessons.append(f"entry_threshold dilonggarkan -> {brain['entry_threshold']:.2f}")

    total = brain["total_wins"] + brain["total_losses"]
    if total >= 10:
        wr = brain["total_wins"] / total
        if wr < 0.4:
            brain["alloc_frac"] = max(0.1, brain["alloc_frac"] * 0.9)
            lessons.append(f"win-rate rendah ({wr:.0%}) -> ukuran posisi dikecilkan")
        elif wr > 0.6:
            brain["alloc_frac"] = min(0.9, brain["alloc_frac"] * 1.05)

    log_experience({
        "t": time.time(), "type": "online", "pnl_pct": pnl_pct,
        "reason": trade.get("reason"), "lessons": lessons,
        "brain": {k: brain[k] for k in ("entry_threshold", "stop_loss_pct", "take_profit_pct", "alloc_frac")},
    })
    return lessons


# ================= 2. BACKTEST EVOLUSIONER (offline learning) =================

def run_backtest(strategies: list[Strategy], df, features, brain: dict,
                 fee: float = 0.003) -> dict:
    """Backtest walk-forward sederhana pada DataFrame OHLCV + fitur.

    Urutan argumen disamakan dengan combined_signal(strategies, features, i):
    (strategies, df, features, brain).

    Threshold entry yang dipakai = rata-rata gerbang adaptif portofolio
    (portfolio_gates), BUKAN angka template — sehingga evolusi bisa belajar
    "berapa ambang yang tepat" bersamaan dengan bobot & parameter strategi.
    """
    from .gates import ENTRY_THR_MIN, ENTRY_THR_MAX   # lokal: hindari siklus impor
    n = len(df)
    # Akuntansi equity-based: eq = kas + nilai pasar posisi. Pembelian memakai
    # fraksi tetap dari EKUITAS (bukan seluruh sisa kas) sehingga tidak ada
    # "uang hantu" yang membuat return backtest membengkak secara artifisial.
    cash, pos_qty, entry_px = 1.0, 0.0, 0.0
    eq_curve, trades = [], []
    thr = float(np.clip(portfolio_gates(strategies)["entry_thr"],
                        ENTRY_THR_MIN, ENTRY_THR_MAX))
    sl, tp = brain["stop_loss_pct"], brain["take_profit_pct"]
    alloc = float(brain.get("alloc_frac", 0.5))
    close = df["close"].values
    atr_series = features["atr_raw"] if "atr_raw" in features else None
    med_atr = float(np.median(atr_series[atr_series > 0])) if atr_series is not None else None

    for i in range(2, n):
        px = close[i]
        if pos_qty > 0:
            chg = px / entry_px - 1
            sig = combined_signal(strategies, features, i)
            if med_atr and entry_px > 0:
                scale = float(np.clip(atr_series[i] / med_atr, 0.5, 3.0))
            else:
                scale = 1.0
            exit_reason = None
            if chg <= -sl * scale:
                exit_reason = "stop_loss"
            elif chg >= tp * scale:
                exit_reason = "take_profit"
            elif sig < -thr * 0.5:
                exit_reason = "signal_exit"
            if exit_reason:
                cash += pos_qty * px * (1 - fee)
                trades.append({"pnl_pct": chg - 2 * fee, "reason": exit_reason})
                pos_qty = 0.0
        else:
            sig = combined_signal(strategies, features, i)
            if sig > thr:
                eq = cash + pos_qty * px
                spend = min(cash, max(0.0, eq * alloc))
                if spend > 1e-9:
                    pos_qty = spend * (1 - fee) / px
                    cash -= spend
                    entry_px = px
        eq = cash + (pos_qty * px if pos_qty else 0.0)
        eq_curve.append(eq)

    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]
    ret = eq_curve[-1] - 1.0 if eq_curve else 0.0
    dd = 0.0
    if eq_curve:
        peak = np.maximum.accumulate(np.array(eq_curve))
        dd = float(np.max(1 - np.array(eq_curve) / peak))
    sharpe = 0.0
    if len(eq_curve) > 10:
        r = np.diff(eq_curve) / np.array(eq_curve[:-1])
        sharpe = float(r.mean() / (r.std() + 1e-9) * np.sqrt(24 * 365))
    fitness = ret * 100 + 20 * (len(wins) / max(1, len(trades))) - dd * 50 + sharpe * 2
    # ---- insentif aktivitas ("bot harus berani trading untuk belajar") ----
    # Varian yang menghasilkan NOL trade tidak memberi AI pengalaman apa pun,
    # jadi dihukum keras; varian dengan sedikit trade diberi bonus kecil agar
    # evolusi TIDAK memilih ambang super longgar demi membanjiri transaksi.
    ntr = len(trades)
    if ntr == 0:
        fitness -= 40.0
    elif ntr < 5:
        fitness += 6.0 * ntr / 5.0
    return {"return_pct": ret * 100, "win_rate": len(wins) / max(1, len(trades)),
            "trades": len(trades), "max_dd_pct": dd * 100, "sharpe": sharpe,
            "entry_thr_used": thr, "fitness": fitness}


def evolve(strategies: list[Strategy], df, features, brain: dict,
           generations: int = 6, pop: int = 4) -> tuple[list[Strategy], dict]:
    """Evolusi portofolio: uji varian bermutasi, pertahankan yang terbaik.

    Ini adalah cara AI 'menulis ulang strateginya sendiri' — ia menghasilkan
    kandidat parameter baru, mengukur kesalahannya lewat backtest, dan hanya
    menyimpan kode konfigurasi yang lebih baik.
    """
    best = strategies
    gates_before = portfolio_gates(best)
    best_res = run_backtest(best, df, features, brain)
    history = [best_res]

    for gen in range(generations):
        candidates = [mutate(best, rate=0.35) for _ in range(pop)]
        for cand in candidates:
            res = run_backtest(cand, df, features, brain)
            history.append(res)
            if res["fitness"] > best_res["fitness"]:
                best, best_res = cand, res
        # setiap generasi, beri peluang mutasi liar (exploration)
        if np.random.random() < 0.3:
            wild = mutate(best, rate=0.6)
            res = run_backtest(wild, df, features, brain)
            if res["fitness"] > best_res["fitness"]:
                best, best_res = wild, res

    # catat fitness per strategi dari kontribusi backtest terakhir
    for s in best:
        s.fitness = round(best_res["fitness"], 3)

    # ---- sinkronisasi dua arah antara otak & gerbang portofolio ----
    # Ambang hasil evolusi (portfolio_gates) dipromosikan ke brain sebagai
    # ambang aktif bot; brain yang lebih berani juga menular ke strategi
    # baru agar benih awal tidak terlalu ketat.
    evolved_thr = float(best_res.get("entry_thr_used", brain["entry_threshold"]))
    brain["entry_threshold"] = round(float(np.clip(
        min(evolved_thr, max(brain["entry_threshold"], ENTRY_THR_MIN)),
        ENTRY_THR_MIN, ENTRY_THR_MAX)), 3)
    g = portfolio_gates(best)
    if g["entry_thr"] < brain["entry_threshold"]:
        brain["entry_threshold"] = round(g["entry_thr"], 3)

    log_experience({"t": time.time(), "type": "evolve", "result": best_res,
                    "generations": generations,
                    "gates_before": gates_before,
                    "gates_after": {k: getattr(best[0], k) for k in
                                    ("entry_thr", "score_min", "signal_floor",
                                     "scalp_strong", "scalp_floor")}
                    if best else {},
                    "brain_entry_threshold": brain["entry_threshold"]})
    return best, best_res
