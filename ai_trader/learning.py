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

import copy
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


def _atomic_write(path: str, obj) -> None:
    """Tulis atomik (tmp + fsync + rename) — mencegah JSON parsial bila
    proses mati di tengah tulis; pembaca tidak pernah melihat file setengah."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_brain(brain: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    _atomic_write(BRAIN_PATH, brain)


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
        # tetap ada (belajar dari rugi) tapi tak boleh mencekik proses belajar.
        # FIX "bot tidak belajar dari 32 trade": sebelumnya setiap loss selalu
        # +0.02 tanpa pandang bulu -> setelah rangkaian loss beruntun, threshold
        # mengunci di cap 0.45 dan penurunan hanya bisa datang dari PROFIT
        # (-0.01) — ratchet naik-turun yang asimetris: bot makin lama makin
        # takut, reject terus, pengalaman berhenti masuk. Sekarang kenaikan
        # mengecil seiring jumlah loss (0.02 -> minimal 0.005), persis seperti
        # manusia yang sudah kapok: tetap waspada, tapi tidak membeku.
        n_loss = brain.get("total_losses", 0)
        bump = max(0.005, 0.02 * (10.0 / max(10.0, n_loss)))
        brain["entry_threshold"] = min(ENTRY_THR_MAX,
                                       round(brain["entry_threshold"] + bump, 3))
        lessons.append(f"entry_threshold dinaikkan +{bump:.3f} "
                       f"-> {brain['entry_threshold']:.3f} (lebih selektif)")
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
        # FIX asimetri keberanian: penurunan threshold kini juga boleh datang
        # dari bukti "rugi kecil terkontrol" (bukan hanya profit). Loss dengan
        # |pnl| <= stop_loss adalah loss yang RENCANANYA BERJALAN — sistem
        # bekerja sebagaimana mestinya, jadi bot tidak boleh makin membeku
        # karenanya. Hanya loss besar (tembus SL / slippage ekstrem) yang
        # menaikkan kewaspadaan. Ini membuat equilibrium threshold turun ke
        # level sehat bahkan saat win-rate masih rendah -> pengalaman terus
        # masuk, pembelajaran tidak berhenti (filosofi: takut-takut kapan belajar).
        thr_relief = (pnl_pct < 0 and abs(pnl_pct) <= brain["stop_loss_pct"] * 1.2
                      and wr < 0.5)
        if pnl_pct > 0 or thr_relief:
            old_thr = brain["entry_threshold"]
            brain["entry_threshold"] = max(ENTRY_THR_MIN,
                                           round(old_thr - 0.005, 3))
            if brain["entry_threshold"] < old_thr - 1e-6:
                tag = ("profit" if pnl_pct > 0
                       else "loss-terkontrol")
                lessons.append(f"{tag} -> entry_threshold dilonggarkan sendiri "
                               f"{old_thr:.3f}->{brain['entry_threshold']:.3f} "
                               "(belajar butuh pengalaman)")
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
                 fee: float = 0.003, slippage: float = 0.0005) -> dict:
    """Backtest walk-forward REALISTIS pada DataFrame OHLCV + fitur.

    Urutan argumen disamakan dengan combined_signal(strategies, features, i):
    (strategies, df, features, brain).

    FIX AUDIT FASE 3 (tiga bug terbukti oleh repro terminal):
    1. LOOK-AHEAD ENTRY: dulu order dieksekusi di close[i] — harga yang sama
       yang dipakai menghitung sinyal bar i, padahal close baru diketahui
       SETELAH bar selesai. Sekarang: sinyal bar i -> eksekusi OPEN bar i+1.
    2. EXIT TIDAK KONSISTEN DENGAN LIVE: dulu exit hanya dicek pada close,
       padahal live memakai ticker intrabar (exit terjadi saat harga MENYENTUH
       level). Bukti repro: 17/200 event SL tersentuh-intrabar tak tereksekusi.
       Sekarang TP/SL/trailing dievaluasi INTRABAR via high/low; bila TP dan SL
       tersentuh pada bar yang sama (urutan tak mungkin diketahui dari OHLCV),
       aturan KONSERVATIF memilih kerugian (SL menang) — bukan klaim tahu urutan.
    3. TRAILING STOP HILANG: kini diperhitungkan (peak berjalan sejak entry),
       konsisten dengan jalur tren live. Exit sinyal memakai ambang TETAP
       SIGNAL_EXIT_FLOOR pada close-bar (sama seperti live) — tidak ikut
       threshold entry (konsistensi Fase 1 P1).
    Biaya dua sisi + slippage kecil dihitung pada kedua arah.

    Threshold entry yang dipakai = rata-rata gerbang adaptif portofolio
    (portfolio_gates), BUKAN angka template — sehingga evolusi bisa belajar
    "berapa ambang yang tepat" bersamaan dengan bobot & parameter strategi.
    """
    from .gates import ENTRY_THR_MIN, ENTRY_THR_MAX   # lokal: hindari siklus impor
    SIGNAL_EXIT_FLOOR = 0.10                          # sama dgn main.py (live)
    n = len(df)
    cash, pos_qty, entry_px = 1.0, 0.0, 0.0
    eq_curve, trades = [], []
    thr = float(np.clip(portfolio_gates(strategies)["entry_thr"],
                        ENTRY_THR_MIN, ENTRY_THR_MAX))
    sl, tp = brain["stop_loss_pct"], brain["take_profit_pct"]
    alloc = float(brain.get("alloc_frac", 0.5))
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    atr_series = features["atr_raw"] if "atr_raw" in features else None
    med_atr = float(np.median(atr_series[atr_series > 0])) if atr_series is not None else None

    pending_entry = False          # sinyal bar i -> beli open bar i+1
    peak_chg = 0.0

    def _exit(px_exit: float, reason: str, i_: int):
        nonlocal cash, pos_qty, entry_px, peak_chg
        px_eff = px_exit * (1 - slippage)
        cash += pos_qty * px_eff * (1 - fee)
        trades.append({"pnl_pct": px_eff / entry_px - 1 - 2 * fee, "reason": reason})
        pos_qty, entry_px, peak_chg = 0.0, 0.0, 0.0

    for i in range(2, n):
        px = close[i]
        # ---- EXIT intrabar pada bar i (bar setelah entry maupun berikutnya) ----
        if pos_qty > 0:
            if med_atr and entry_px > 0:
                scale = float(np.clip(atr_series[i] / med_atr, 0.5, 3.0))
            else:
                scale = 1.0
            sl_lvl, tp_lvl = sl * scale, tp * scale
            hi_chg = high[i] / entry_px - 1
            lo_chg = low[i] / entry_px - 1
            tp_hit, sl_hit = hi_chg >= tp_lvl, lo_chg <= -sl_lvl
            if tp_hit and sl_hit:
                # urutan intrabar TAKSIRAN-TERBATAS dari OHLCV -> konservatif:
                # anggap kerugian lebih dulu menyentuh (SL menang). Jangan pernah
                # mengklaim tahu urutan sebenarnya.
                _exit(entry_px * (1 - sl_lvl), "stop_loss", i)
            elif sl_hit:
                _exit(entry_px * (1 - sl_lvl), "stop_loss", i)
            elif tp_hit:
                _exit(entry_px * (1 + tp_lvl), "take_profit", i)
            else:
                # trailing (konsisten live): kunci profit setelah peak >= 60% TP
                new_peak = max(peak_chg, hi_chg)
                trail_sl = max(sl, 0.02)
                if peak_chg >= tp_lvl * 0.6 and lo_chg < new_peak - trail_sl:
                    _exit(entry_px * (1 + new_peak - trail_sl), "trailing_stop", i)
                else:
                    peak_chg = new_peak
                    sig = combined_signal(strategies, features, i)
                    if sig < -SIGNAL_EXIT_FLOOR:      # exit-sinyal pada close-bar
                        _exit(px, "signal_exit", i)
        # ---- ENTRY tertunda: sinyal bar i-1 dieksekusi di open bar i ----
        if pos_qty == 0 and pending_entry:
            ep = open_[i] * (1 + slippage)
            eq = cash + pos_qty * px
            spend = min(cash, max(0.0, eq * alloc))
            if spend > 1e-9:
                pos_qty = spend * (1 - fee) / ep
                cash -= spend
                entry_px = ep
                peak_chg = 0.0
        pending_entry = False
        if pos_qty == 0:
            sig = combined_signal(strategies, features, i)
            if sig > thr:
                pending_entry = True     # keputusan malam -> esok pagi (open)
        eq = cash + (pos_qty * px if pos_qty else 0.0)
        eq_curve.append(eq)

    # posisi terbuka di akhir backtest ditutup pada close terakhir (mark-to-market
    # jujur, bukan dibiarkan menguap dari kurva)
    if pos_qty > 0:
        _exit(float(close[-1]), "end_of_test", n - 1)
        eq_curve[-1] = cash

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
    """Evolusi portofolio dengan pemisahan TRAIN/VALIDASI berdasarkan WAKTU.

    FIX AUDIT FASE 4 (overfitting & seleksi tanpa OOS):
    * Data dipecah temporal: 70% pertama = training (mutasi & seleksi),
      30% terakhir = out-of-sample VALIDASI yang TIDAK PERNAH dilihat selama
      seleksi. Kandidat terbaik dipilih di train, lalu HARUS membuktikan diri
      di validasi sebelum diterapkan.
    * Validasi gagal (return-OOS lebih buruk dari baseline > toleransi, atau
      nol trade) -> strategi lama DIPERTAHANKAN; parameter hasil evolusi tidak
      diterapkan agresif (aturan "jangan ganti hanya karena fitness naik
      sedikit" kini ditegakkan lewat margin).
    * Perbaikan crash terbukti: `brain["entry_threshold"]` diakses tanpa guard
      saat best_res tidak memuat 'entry_thr_used' -> KeyError (repro terminal).
    """
    n = len(df)
    split = max(120, int(n * 0.7)) if n >= 180 else n     # tanpa split bila data pendek
    if split < n:
        tr_df = df.iloc[:split].reset_index(drop=True)
        va_df = df.iloc[split - 60:].reset_index(drop=True)   # overlap 60 bar: warm-start fitur
        try:
            tr_feat = features.iloc[:split].reset_index(drop=True)
            va_feat = features.iloc[split - 60:].reset_index(drop=True)
        except Exception:
            tr_feat, va_feat = features, features
    else:
        tr_df, va_df = df, None
        tr_feat = va_feat = features

    def _bt(strat, d, f):
        return run_backtest(strat, d, f, brain)

    best = strategies
    gates_before = portfolio_gates(best)
    best_res = _bt(best, tr_df, tr_feat)
    base_val = _bt(copy.deepcopy(best), va_df, va_feat) if va_df is not None else best_res
    history = [best_res]

    for gen in range(generations):
        candidates = [mutate(best, rate=0.35) for _ in range(pop)]
        for cand in candidates:
            res = _bt(cand, tr_df, tr_feat)
            history.append(res)
            if res["fitness"] > best_res["fitness"]:
                best, best_res = cand, res
        if np.random.random() < 0.3:
            wild = mutate(best, rate=0.6)
            res = _bt(wild, tr_df, tr_feat)
            if res["fitness"] > best_res["fitness"]:
                best, best_res = wild, res

    # ---- MARGIN penerimaan: peningkatan fitness harus MELEBIHI ambang noise
    # (bukan sekadar >) — mencegah strategi diganti karena fluktuasi kecil.
    FITNESS_MARGIN = 0.5
    accepted = best_res["fitness"] >= _bt(strategies, tr_df, tr_feat)["fitness"] + FITNESS_MARGIN

    # ---- VALIDASI OUT-OF-SAMPLE: kandidat juara diuji pada data masa depan
    # yang tidak pernah dilihat seleksi. Gagal -> rollback penuh ke seed.
    val_res = None
    if va_df is not None and accepted:
        val_res = _bt(copy.deepcopy(best), va_df, va_feat)
        val_ok = (val_res["trades"] > 0
                  and val_res["return_pct"] >= base_val["return_pct"] - 1.0
                  and val_res["max_dd_pct"] <= max(base_val["max_dd_pct"] + 5.0, 15.0))
        if not val_ok:
            log_experience({"t": time.time(), "type": "evolve_rejected_oos",
                            "train_fitness": best_res["fitness"],
                            "oos": {k: round(float(val_res[k]), 3)
                                    for k in ("return_pct", "trades", "max_dd_pct")}})
            best, best_res, accepted = strategies, best_res, False
            val_res = {"rejected": True, **{k: round(float(val_res[k]), 3) for k in
                                            ("return_pct", "trades", "max_dd_pct")}}

    # catat fitness per strategi dari backtest terakhir yang DITERIMA
    for s in best:
        s.fitness = round(best_res["fitness"], 3)

    # ---- sinkronisasi dua arah antara otak & gerbang portofolio ----
    evolved_thr = float(best_res.get("entry_thr_used",
                                     brain.get("entry_threshold", ENTRY_THR_MIN)))
    brain.setdefault("entry_threshold", evolved_thr)
    brain["entry_threshold"] = round(float(np.clip(
        min(evolved_thr, max(brain["entry_threshold"], ENTRY_THR_MIN)),
        ENTRY_THR_MIN, ENTRY_THR_MAX)), 3)
    g = portfolio_gates(best)
    if g["entry_thr"] < brain["entry_threshold"]:
        brain["entry_threshold"] = round(g["entry_thr"], 3)

    log_experience({"t": time.time(), "type": "evolve", "result": best_res,
                    "generations": generations, "accepted": accepted,
                    "train_split": split, "oos_validation": val_res,
                    "gates_before": gates_before,
                    "gates_after": {k: getattr(best[0], k) for k in
                                    ("entry_thr", "score_min", "signal_floor",
                                     "scalp_strong", "scalp_floor")}
                    if best else {},
                    "brain_entry_threshold": brain["entry_threshold"]})
    # laporkan hasil VALIDASI bila ada (jujur: angka yang dipakai untuk promosi)
    report = dict(best_res)
    if val_res is not None and not val_res.get("rejected"):
        report.update({"oos_return_pct": val_res["return_pct"],
                       "oos_trades": val_res["trades"],
                       "oos_max_dd_pct": val_res["max_dd_pct"]})
    report["accepted"] = accepted
    return best, report
