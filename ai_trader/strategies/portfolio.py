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


STRATEGY_FNS = {
    "sig_rsi_cross": sig_rsi_cross,
    "sig_macd_momentum": sig_macd_momentum,
    "sig_ema_trend": sig_ema_trend,
    "sig_bollinger_reversion": sig_bollinger_reversion,
    "sig_stochastic": sig_stochastic,
    "sig_candle_reversal": sig_candle_reversal,
    "sig_volume_confirmation": sig_volume_confirmation,
    "sig_vwap_deviation": sig_vwap_deviation,
}


def default_portfolio() -> list[Strategy]:
    return [
        Strategy("rsi_reversal", "indicator", {"overbought": 70, "oversold": 30}, "sig_rsi_cross"),
        Strategy("macd_momentum", "indicator", {"sensitivity": 10.0}, "sig_macd_momentum"),
        Strategy("ema_trend", "indicator", {"sensitivity": 8.0}, "sig_ema_trend"),
        Strategy("bollinger_reversion", "indicator", {"band": 0.9}, "sig_bollinger_reversion"),
        Strategy("stochastic_os", "indicator", {"overbought": 80, "oversold": 20}, "sig_stochastic"),
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fp:
        json.dump([s.to_dict() for s in strategies], fp, indent=2)


def load_portfolio(path: str = STATE_PATH) -> list[Strategy]:
    if os.path.exists(path):
        try:
            with open(path) as fp:
                return [Strategy.from_dict(d) for d in json.load(fp)]
        except Exception:
            pass
    return default_portfolio()


PARAM_BOUNDS = {
    "overbought": (60, 90), "oversold": (10, 40),
    "sensitivity": (1.0, 30.0), "band": (0.5, 1.5),
    "strength": (0.1, 2.0), "spike": (1.1, 4.0),
}


def mutate(strategies: list[Strategy], rate: float = 0.3, rng: random.Random | None = None) -> list[Strategy]:
    """Mutasi parameter & bobot strategi — mekanisme 'self-modifying' AI.

    AI tidak menulis ulang file .py secara sembarangan; ia berevolusi pada
    ruang parameter/bobot yang tervalidasi (disimpan di strategies.json),
    sehingga aman dari crash/injeksi kode.
    """
    rng = rng or random.Random()
    new = copy.deepcopy(strategies)
    for s in new:
        if rng.random() < rate:
            s.weight = float(np.clip(s.weight * rng.uniform(0.7, 1.3), 0.05, 3.0))
        for k, v in list(s.params.items()):
            lo, hi = PARAM_BOUNDS.get(k, (v * 0.5, v * 2.0 + 1e-9))
            if rng.random() < rate:
                s.params[k] = round(float(np.clip(v * rng.uniform(0.8, 1.2), lo, hi)), 3)
    # Kadang AI "menulis" strategi baru dari template yang ada
    if rng.random() < 0.15 and len(new) < 14:
        base = rng.choice(new)
        child = copy.deepcopy(base)
        child.name = f"{base.name}_v{rng.randint(100, 999)}"
        child.weight = 0.5
        child.fitness = 0.0
        for k in child.params:
            child.params[k] = round(child.params[k] * rng.uniform(0.7, 1.3), 3)
        new.append(child)
    return new
