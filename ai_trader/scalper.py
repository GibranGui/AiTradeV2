"""Scalper — jalur trading jangka pendek (15 menit) di samping bot tren 1 jam.

MENGAPA MODUL INI ADA (diagnosis nyata, bukan asumsi):
  1. Gerbang entry tren lama menuntut skor>=0.62 DAN uptrend(mom20>0 & EMA>0)
     DAN sinyal>=0.15 secara BERSAMAAN -> pada candle 1h, event ini langka;
     24 jam penuh Actions = 0 transaksi (dibuktikan log [FUNNEL] siklus).
  2. Loop Actions berjalan ~setiap 15-30 menit, tetapi scanner hanya membaca
     candle 60 menit -> mayoritas run menganalisis bar yang SAMA persis
     (bar belum berganti) sehingga tidak ada informasi baru sama sekali.
     Candle 15m berganti tiap 15 menit = tepat frekuensi loop -> setiap run
     kini membawa data baru.

Desain aman:
  * Data WAJIB candle asli (binance-idr/indodax); sintetis -> ditolak + dicatat.
  * Tidak ada sinyal palsu: semua penolakan dicatat lewat diag dengan nilai
    aktual vs ambang (format [REJECT]).
  * Exit scalping ketat: TP/SL berbasis ATR per-bar, max_hold, trailing,
    dan exit saat sinyal berbalik — semua tetap memakai PaperBroker yang sama
    (fee 0,3% dihitung, posisi dicatat ke state, learn_from_trade berjalan).
  * Risiko terpisah dari brain tren (tidak mengubah SL/TP strategi utama).
  * Tetap PAPER ONLY — tidak menyentuh live_broker.
"""

from __future__ import annotations

import time

import numpy as np

from .features import build_features
from .market_data import fetch_ohlcv_or_synthetic
from .strategies.portfolio import combined_signal, normalize_signal_features

# --- konfigurasi scalping (terbukti dari backtest lokal, lihat laporan) ---
SCALP_INTERVAL_MIN = 15          # candle 15 menit
SCALP_LIMIT = 300                # ~3 hari data 15m
SCALP_SIGNAL_FLOOR = 0.08        # jalur B: konfluensi minimal + mom12>0 + EMA>0
                                 #   (dilonggarkan dari 0.10 — user minta gate
                                 #   lebih terbuka; tetap dijaga mom12+EMA+volume)
SCALP_MAX_CHG24H = 18.0          # hindari koin yang sedang pump ekstrem
SCALP_MIN_VOL_IDR = 500_000_000  # likuiditas minimal (sama dgn universe)
SCALP_MAX_POSITIONS = 2          # pagar risiko: maks 2 slot scalp aktif
SCALP_STRONG_SIGNAL = 0.18       # jalur A: sinyal cukup kuat -> lolos tanpa
                                 #   syarat momentum (edge terkuat tetap >=0.20)
                                 #   (n=118, win-rate 53%, rata2 +0.60%/4 bar)
SCALP_MOM_WINDOW = 12             # momentum dihitung per jam (12 bar 15m),
                                 #   BUKAN 5 bar — mom5 terbukti noise: gate
                                 #   sig>=0.2&mom5>0 justru menurunkan edge.
SCALP_TAKE_PROFIT_PCT = 0.03     # TP 3% — hasil optimal backtest TP/SL live
SCALP_STOP_LOSS_PCT = 0.012      # SL 1,2% (rasio R:R 2.5:1)
SCALP_TRAIL_GIVEBACK = 0.006     # trailing: kunci profit jika turun 0,6% dr puncak
SCALP_MAX_HOLD_SEC = 3 * 60 * 60 # time-stop 3 jam (12 bar 15m; horizon terbaik)


def _mom(close: np.ndarray, bars: int) -> float:
    j = max(0, len(close) - bars)
    return float(close[-1] / max(1e-9, close[j]) - 1)


class Scalper:
    """Scanner khusus timeframe pendek + manajemen posisi scalp."""

    def __init__(self, strategies, interval_min: int = SCALP_INTERVAL_MIN,
                 limit: int = SCALP_LIMIT, cache_sec: int = 240):
        self.strategies = strategies
        self.interval_min = interval_min
        self.limit = limit
        self.cache_sec = cache_sec
        self._cache: dict[str, tuple[float, dict]] = {}

    # ---------- analisis satu kandidat ----------
    def analyze(self, coin: dict, diag=None) -> dict | None:
        pair = coin["pair"]
        hit = self._cache.get(pair)
        if hit and time.time() - hit[0] < self.cache_sec:
            return hit[1]
        try:
            df, is_live = fetch_ohlcv_or_synthetic(
                pair, period=str(self.interval_min), limit=self.limit)
        except Exception as e:
            if diag:
                diag.data_bad(coin["symbol"], "scalp_fetch_exception", str(e)[:80])
            return None
        if df is None or len(df) < 60 or not is_live:
            if diag:
                nbar = 0 if df is None else len(df)
                why = ("sintetis" if (df is not None and nbar >= 60 and not is_live)
                       else f"bar={nbar}<60")
                diag.data_bad(coin["symbol"], "scalp_candle_data_incomplete", why)
            return None
        feats = normalize_signal_features(build_features(df))
        i = len(feats) - 1
        sig = combined_signal(self.strategies, feats, i)
        close = feats["close"].to_numpy(dtype=float)
        m5, m20 = _mom(close, 5), _mom(close, 20)
        m12 = _mom(close, SCALP_MOM_WINDOW)   # momentum 1 jam = basis gerbang
        # atr_pct tersimpan ter-skala [-1..1]; konversi balik ke fraksi nyata
        atr_pct = max(0.001, (float(feats["atr_pct"][i]) + 1.0) / 200.0)
        ema_ok = float(feats["ema_trend"][i]) > 0
        vol_sup = float(np.clip(float(feats["obv_slope"][i]) * 0.5
                                + float(feats["vol_sma_ratio"][i]) * 0.25, -1, 1))
        result = {
            "pair": pair, "symbol": coin["symbol"], "signal": sig,
            "mom5": m5, "mom20": m20, "mom12": m12, "atr_pct": atr_pct,
            "ema_ok": bool(ema_ok), "vol_support": vol_sup,
            "last_close": float(close[-1]),
            "chg24h_pct": coin["chg24h_pct"], "vol_idr_24h": coin["vol_idr_24h"],
            "features": {k: float(v) for k, v in feats.iloc[i].items()},
        }
        self._cache[pair] = (time.time(), result)
        return result

    # ---------- gerbang peluang scalp (AMBANG ADAPTIF MILIK AI) ----------
    def scalp_opportunities(self, results: list[dict], diag=None,
                            strong_thr: float | None = None,
                            floor_thr: float | None = None) -> list[dict]:
        """Filter untuk horizon 15m. Ambang default = konstanta modul (fallback);
        loop trading mengirim `strong_thr`/`floor_thr` hasil evolusi AI sehingga
        bot menentukan sendiri seberani apa iascalping. Setiap penolakan dicatat
        lengkap (simbol, tahap, alasan, nilai aktual vs ambang)."""
        if strong_thr is None:
            strong_thr = SCALP_STRONG_SIGNAL
        if floor_thr is None:
            floor_thr = SCALP_SIGNAL_FLOOR
        out = []
        for r in results:
            sym = r["symbol"]
            if r["chg24h_pct"] > SCALP_MAX_CHG24H:
                if diag:
                    diag.reject(sym, "scalp_filter", "pump_extreme",
                                actual=f"chg24h={r['chg24h_pct']:+.1f}%",
                                required=f"<={SCALP_MAX_CHG24H}%")
                continue
            # Dua jalur lolos (berdasarkan backtest data live):
            #  A) sinyal KUAT >= strong_thr — edge terkuat, lolos tanpa syarat lain;
            #  B) sinyal sedang + momentum 1-jam positif + EMA naik.
            strong = r["signal"] >= strong_thr
            mid_ok = (r["signal"] >= floor_thr
                      and r.get("mom12", r["mom5"]) > 0 and r["ema_ok"])
            if not (strong or mid_ok):
                if diag:
                    req = f"signal>={strong_thr:.2f} ATAU (>={floor_thr:.2f} & mom12>0 & ema>0)"
                    diag.reject(sym, "scalp_filter", "signal_below_floor",
                                actual=f"signal={r['signal']:+.3f} mom12={r.get('mom12', 0):+.3%} ema_ok={r['ema_ok']}",
                                required=req)
                continue
            if r["vol_support"] <= 0:
                # volume harus mendukung (OBV naik / vol di atas SMA) —
                # filter ekstra yang membayar kelonggaran signal floor
                if diag:
                    diag.reject(sym, "scalp_filter", "volume_not_supporting",
                                actual=f"vol_support={r['vol_support']:+.3f}",
                                required=">0")
                continue
            out.append(r)
        # prioritas: likuiditas terbesar dulu (mudah exit), lalu kekuatan sinyal
        out.sort(key=lambda x: (-x["vol_idr_24h"], -x["signal"]))
        return out

    # ---------- scan ----------
    def scan(self, universe: list[dict], deep_n: int = 10, diag=None) -> list[dict]:
        cands = [u for u in universe
                 if u["vol_idr_24h"] >= SCALP_MIN_VOL_IDR
                 and u["chg24h_pct"] <= SCALP_MAX_CHG24H]
        cands.sort(key=lambda u: -u["vol_idr_24h"])
        cands = cands[:deep_n]
        if diag:
            diag.deep_start(len(cands))
        results = []
        for c in cands:
            r = self.analyze(c, diag=diag)
            if r:
                if diag:
                    diag.data_ok()
                results.append(r)
            time.sleep(0.1)
        return results


def scalp_exit_decision(pos, price: float, sig_now: float | None) -> str | None:
    """Keputusan exit khusus scalp. Mengembalikan alasan atau None.

    Prioritas: stop_loss -> take_profit (ATR-adaptif) -> trailing -> time_stop
    -> signal_exit. Semua ambang batas konservatif (rugi dibatasi sebelum profit)."""
    chg = price / pos.entry_price - 1
    atr = float(pos.entry_features.get("_scalp_atr_pct", 0.0)) or None
    tp = SCALP_TAKE_PROFIT_PCT
    sl = SCALP_STOP_LOSS_PCT
    if atr:                                   # sesuaikan dgn volatilitas bar
        tp = float(np.clip(tp * (1 + atr * 10), tp, tp * 2.0))
        sl = float(np.clip(sl * (1 + atr * 10), sl, sl * 2.0))
    peak = pos.entry_features.get("_peak_chg", 0.0)
    if chg > peak:
        pos.entry_features["_peak_chg"] = chg
        peak = chg
    if chg <= -sl:
        return "scalp_stop_loss"
    if chg >= tp:
        return "scalp_take_profit"
    if peak >= tp * 0.6 and chg < peak - SCALP_TRAIL_GIVEBACK:
        return "scalp_trailing_stop"
    if chg > 0 and time.time() - pos.entry_time > SCALP_MAX_HOLD_SEC:
        return "scalp_time_stop"      # kunci profit menua; rugi dibiarkan
                                      # bekerja sampai SL/TP (sesuai backtest)
    if sig_now is not None and sig_now < -0.05:
        return "scalp_signal_exit"
    return None
