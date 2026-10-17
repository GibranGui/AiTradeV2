"""Strategi modular: setiap strategi punya parameter yang bisa dimutasi oleh AI.

Setiap strategi mengembalikan sinyal dalam rentang [-1, 1]:
   +1 = beli kuat, -1 = jual kuat, 0 = netral.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from ..gates import ENTRY_THR_MIN, ENTRY_THR_MAX, SCORE_THR_MIN, gate_snapshot


@dataclass
class Strategy:
    name: str
    group: str                      # indicator | candle | volume
    params: dict                    # parameter yang dapat dimutasi AI
    signal_fn_name: str             # nama fungsi di STRATEGY_FNS (tetap, tidak dieksekusi sebagai kode)
    weight: float = 1.0             # bobot gabungan — AI belajar menyesuaikan ini
    fitness: float = 0.0            # skor hasil backtest terakhir
    wins: int = 0
    losses: int = 0
    # ---- gerbang adaptif MILIK AI (bukan template manusia) ----
    entry_thr: float = 0.20         # ambang sinyal untuk strategi ini
    score_min: float = 0.45         # ambang skor peluang tren (jalur tren)
    signal_floor: float = 0.08      # konfirmasi arah minimal (jalur tren)
    scalp_strong: float = 0.15      # jalur A scalp: sinyal kuat tanpa syarat lain
    scalp_floor: float = 0.06       # jalur B scalp: sinyal sedang + konfluensi

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Strategy(**d)


# ---------- Fungsi sinyal (murni, aman, tanpa eval) ----------

def sig_rsi_cross(f, i, p):
    """Beli saat RSI oversold, jual saat overbought. p: threshold."""
    rsi = f["rsi"][i]  # tersimpan ter-normalisasi (-1..1), 0 = netral(50)
    ob = p["overbought"] / 50 - 1     # mis. 70 -> 0.4
    os_ = p["oversold"] / 50 - 1      # mis. 30 -> -0.4
    if rsi < os_:
        return 1.0
    if rsi > ob:
        return -1.0
    return 0.0


def sig_macd_momentum(f, i, p):
    h = f["macd_hist"][i]
    k = p["sensitivity"]
    return float(np.tanh(h * k))


def sig_ema_trend(f, i, p):
    t = f["ema_trend"][i]
    return float(np.clip(t * p["sensitivity"], -1, 1))


def sig_bollinger_reversion(f, i, p):
    bb = f["bb_pos"][i]
    if bb < -p["band"]:
        return 1.0
    if bb > p["band"]:
        return -1.0
    return 0.0


def sig_stochastic(f, i, p):
    s = f["stoch_k"][i]
    if s < p["oversold"] / 50 - 1:
        return 1.0
    if s > p["overbought"] / 50 - 1:
        return -1.0
    return 0.0


def sig_candle_reversal(f, i, p):
    bull = f["engulfing_bull"][i] + f["hammer"][i] + f["three_white"][i] + f["marubozu_bull"][i]
    bear = f["engulfing_bear"][i] + f["shooting_star"][i] + f["three_black"][i] + f["marubozu_bear"][i]
    strength = p["strength"]
    return float(np.clip((bull - bear) * strength, -1, 1))


def sig_volume_confirmation(f, i, p):
    """Volume spike memperkuat arah momentum (OBV slope)."""
    ratio = f["vol_sma_ratio"][i]
    obv = f["obv_slope"][i]
    if ratio > p["spike"]:
        return float(np.clip(obv, -1, 1))
    return 0.0


def sig_vwap_deviation(f, i, p):
    dev = f["vwap_dev"][i]
    k = p["sensitivity"]
    return float(np.clip(-dev * k, -1, 1))  # reversion ke VWAP


def _zscore(s, lookback: int = 60) -> "pd.Series":
    """Standarisasi rolling (mean≈0, std≈1) agar distribusi sinyal seimbang."""
    m = s.rolling(lookback, min_periods=10).mean()
    sd = s.rolling(lookback, min_periods=10).std().replace(0, np.nan)
    return ((s - m) / sd).fillna(0).clip(-4, 4) / 4   # ~[-1, 1]


def sig_breakout(f, i, p):
    """Momentum breakout: harga menembus high N-bar terakhir dengan volume di
    rata-rata. p: lookback (jumlah bar), vol_confirm (rasio volume minimum)."""
    lb = max(2, int(p.get("lookback", 20)))
    if i < lb + 1:
        return 0.0
    closes = f["close"]
    vols = f["volume"]
    prev_high = float(closes[max(0, i - lb):i].max())
    prev_low = float(closes[max(0, i - lb):i].min())
    px = float(closes[i])
    vol_ok = vols[i] >= p.get("vol_confirm", 1.0) * max(1e-9, float(vols[max(0, i - lb):i].mean()))
    if px > prev_high and vol_ok:
        return 1.0
    if px < prev_low and vol_ok:
        return -1.0
    return 0.0


def sig_ma_cross(f, i, p):
    """Golden/death cross: EMA cepat vs EMA lambat (trend-following)."""
    fast = f["ema_fast"]
    slow = f["ema_slow"]
    diff = (fast[i] - slow[i]) / max(1e-9, abs(slow[i]))
    return float(np.clip(diff * p.get("sensitivity", 400.0), -1, 1))


def normalize_signal_features(f: "pd.DataFrame") -> "pd.DataFrame":
    """Z-score fitur kontinu yang tidak simetris distribusinya (macd/ema/vwap/obv)
    supaya sinyal gabungan benar-benar bergerak di sekitar nol dan threshold
    entry bisa terlampaui. Fitur diskrit (pola candle) & yang sudah simetris
    (rsi, bb_pos, stoch_k) tidak diubah."""
    for col in ("macd_hist", "ema_trend", "vwap_dev", "obv_slope"):
        if col in f.columns:
            f[col] = _zscore(pd.Series(f[col], index=f.index)).values
    return f


STRATEGY_FNS = {
    "sig_rsi_cross": sig_rsi_cross,
    "sig_macd_momentum": sig_macd_momentum,
    "sig_ema_trend": sig_ema_trend,
    "sig_bollinger_reversion": sig_bollinger_reversion,
    "sig_stochastic": sig_stochastic,
    "sig_candle_reversal": sig_candle_reversal,
    "sig_volume_confirmation": sig_volume_confirmation,
    "sig_vwap_deviation": sig_vwap_deviation,
    "sig_breakout": sig_breakout,
    "sig_ma_cross": sig_ma_cross,
}


def default_portfolio() -> list[Strategy]:
    return [
        Strategy("rsi_reversal", "indicator", {"overbought": 70, "oversold": 30}, "sig_rsi_cross"),
        Strategy("macd_momentum", "indicator", {"sensitivity": 10.0}, "sig_macd_momentum"),
        Strategy("ema_trend", "indicator", {"sensitivity": 8.0}, "sig_ema_trend"),
        Strategy("bollinger_reversion", "indicator", {"band": 0.9}, "sig_bollinger_reversion"),
        Strategy("stochastic_os", "indicator", {"overbought": 80, "oversold": 20}, "sig_stochastic"),
        Strategy("donchian_breakout", "indicator", {"lookback": 20, "vol_confirm": 1.0}, "sig_breakout"),
        Strategy("ma_cross_trend", "indicator", {"sensitivity": 400.0}, "sig_ma_cross"),
        Strategy("candle_reversal", "candle", {"strength": 0.5}, "sig_candle_reversal"),
        Strategy("volume_spike", "volume", {"spike": 1.8}, "sig_volume_confirmation"),
        Strategy("vwap_reversion", "volume", {"sensitivity": 2.0}, "sig_vwap_deviation"),
    ]


def combined_signal(strategies: list[Strategy], features, i: int) -> float:
    """Sinyal gabungan berbobot semua strategi, dinormalisasi ke [-1, 1]."""
    num, den = 0.0, 0.0
    for s in strategies:
        fn = STRATEGY_FNS.get(s.signal_fn_name)
        if fn is None:
            continue
        sig = fn(features, i, s.params)
        num += s.weight * sig
        den += abs(s.weight)
    return num / den if den > 0 else 0.0


# ---------- Persistensi & mutasi (AI mengubah "script"-nya sendiri) ----------

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "strategies.json")


def save_portfolio(strategies: list[Strategy], path: str = STATE_PATH):
    """Tulis atomik (tmp + fsync + rename) — lihat catatan audit Fase 6."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump([s.to_dict() for s in strategies], fp, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def load_portfolio(path: str = STATE_PATH) -> list[Strategy]:
    if os.path.exists(path):
        try:
            with open(path) as fp:
                rows = json.load(fp)
            # migrasi state lama (belum punya field gerbang adaptif):
            # Strategy(**d) menolak kunci asing & kehilangan default baru,
            # jadi saring hanya ke field yang dikenal dataclass.
            known = {f.name for f in __import__("dataclasses").fields(Strategy)}
            return [Strategy(**{k: v for k, v in d.items() if k in known})
                    for d in rows]
        except Exception:
            pass
    return default_portfolio()


def portfolio_gates(strategies: list[Strategy]) -> dict:
    """Ambang AKTIF portofolio = rata-rata bobot gerbang tiap strategi.
    Inilah 'keberanian' yang dipilih AI sendiri — bukan template manusia."""
    tw = sum(max(0.05, s.weight) for s in strategies) or 1.0
    g = {}
    for key in ("entry_thr", "score_min", "signal_floor",
                "scalp_strong", "scalp_floor"):
        g[key] = float(sum(max(0.05, s.weight) * getattr(s, key)
                           for s in strategies) / tw)
    return g


PARAM_BOUNDS = {
    "overbought": (60, 90), "oversold": (10, 40),
    "sensitivity": (1.0, 30.0), "band": (0.5, 1.5),
    "strength": (0.1, 2.0), "spike": (1.1, 4.0),
    "lookback": (10, 55), "vol_confirm": (0.8, 2.5),
}
# sensitivity khusus untuk sig_ma_cross memakai skala berbeda (rasio harga)
def _bounds_for(key: str, strategy: "Strategy", v: float):
    if key == "sensitivity" and strategy.signal_fn_name == "sig_ma_cross":
        return (100.0, 2000.0)
    return PARAM_BOUNDS.get(key, (v * 0.5, v * 2.0 + 1e-9))


GATE_BOUNDS = {          # rentang gerbang adaptif — batas aman, BUKAN nilai tetap
    "entry_thr":    (ENTRY_THR_MIN, ENTRY_THR_MAX),
    "score_min":    (SCORE_THR_MIN, 0.62),
    "signal_floor": (0.0, 0.30),
    "scalp_strong": (0.10, 0.40),
    "scalp_floor":  (0.04, 0.30),
}


def mutate(strategies: list[Strategy], rate: float = 0.3, rng: random.Random | None = None) -> list[Strategy]:
    """Mutasi parameter, bobot, DAN gerbang strategi — mekanisme 'self-modifying' AI.

    AI tidak menulis ulang file .py secara sembarangan; ia berevolusi pada
    ruang parameter/bobot/ambang yang tervalidasi (disimpan di strategies.json),
    sehingga aman dari crash/injeksi kode. Gerbang entry kini ikut bermutasi:
    ambang bukan lagi template manusia — AI menemukannya lewat backtest
    (fitness memberi penalti keras pada varian nol-trade, lihat learning.evolve).
    """
    rng = rng or random.Random()
    new = copy.deepcopy(strategies)
    for s in new:
        if rng.random() < rate:
            s.weight = float(np.clip(s.weight * rng.uniform(0.7, 1.3), 0.05, 3.0))
        for k, v in list(s.params.items()):
            lo, hi = _bounds_for(k, s, v)
            if rng.random() < rate:
                s.params[k] = round(float(np.clip(v * rng.uniform(0.8, 1.2), lo, hi)), 3)
        # ---- mutasi gerbang adaptif milik AI ----
        for gk, (glo, ghi) in GATE_BOUNDS.items():
            if rng.random() < rate:
                cur = getattr(s, gk)
                setattr(s, gk, round(float(np.clip(cur * rng.uniform(0.75, 1.35), glo, ghi)), 3))
    # Kadang AI "menulis" strategi baru dari template yang ada
    if rng.random() < 0.15 and len(new) < 14:
        base = rng.choice(new)
        child = copy.deepcopy(base)
        child.name = f"{base.name}_v{rng.randint(100, 999)}"
        child.weight = 0.5
        child.fitness = 0.0
        for k in child.params:
            child.params[k] = round(child.params[k] * rng.uniform(0.7, 1.3), 3)
        for gk, (glo, ghi) in GATE_BOUNDS.items():
            setattr(child, gk, round(float(np.clip(getattr(child, gk) * rng.uniform(0.7, 1.3), glo, ghi)), 3))
        new.append(child)
    return new
