"""Ekstraksi fitur: indikator teknikal, bentuk/pola candlestick, dan analisis volume."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------- Indikator teknikal ----------------

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(close: pd.Series):
    line = ema(close, 12) - ema(close, 26)
    sig = line.ewm(span=9, adjust=False).mean()
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int = 20):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return mid, mid + 2 * sd, mid - 2 * sd


def stochastic(df: pd.DataFrame, k_period: int = 14):
    lo = df["low"].rolling(k_period).min()
    hi = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return k.fillna(50)


# ---------------- Bentuk candlestick ----------------

def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Deteksi pola-pola candle utama. Menghasilkan kolom biner [-1..1]."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    avg_body = body.rolling(20).mean()

    p = pd.DataFrame(index=df.index)
    bull = c > o
    bear = c < o

    p["engulfing_bull"] = ((bull) & (c.shift() < o.shift()) & (body > (o - c).shift())).astype(int)
    p["engulfing_bear"] = ((bear) & (c.shift() > o.shift()) & (body > (c - o).shift())).astype(int)
    p["hammer"] = ((lower > 2 * body) & (upper < body + 1e-9)).astype(int)
    p["shooting_star"] = ((upper > 2 * body) & (lower < body + 1e-9)).astype(int)
    p["doji"] = (body < 0.1 * avg_body).astype(int)
    p["marubozu_bull"] = ((bull) & (rng > 0) & (upper < 0.05 * rng) & (lower < 0.05 * rng)).astype(int)
    p["marubozu_bear"] = ((bear) & (rng > 0) & (upper < 0.05 * rng) & (lower < 0.05 * rng)).astype(int)
    p["three_white"] = (bull & bull.shift() & bull.shift(2) &
                        (c > c.shift()) & (c.shift() > c.shift(2))).astype(int)
    p["three_black"] = (bear & bear.shift() & bear.shift(2) &
                        (c < c.shift()) & (c.shift() < c.shift(2))).astype(int)
    p["inside_bar"] = ((h <= h.shift()) & (l >= l.shift())).astype(int)
    return p.fillna(0)


# ---------------- Analisis volume ----------------

def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    v = df["volume"]
    out = pd.DataFrame(index=df.index)
    out["vol_sma_ratio"] = (v / v.rolling(20).mean()).clip(0, 10).fillna(1)
    direction = np.sign(df["close"].diff()).fillna(0)
    ob = (direction * v).cumsum()
    out["obv_slope"] = (ob.diff(5) / (v.rolling(5).mean() + 1)).clip(-5, 5).fillna(0)
    upv = v.where(direction > 0, 0.0)
    dnv = v.where(direction < 0, 0.0)
    out["vwap_dev"] = (((df["close"] * v).cumsum() / v.cumsum().replace(0, np.nan))
                       .pipe(lambda x: (df["close"] - x) / df["close"]) * 100).fillna(0)
    out["volatility_pct"] = (df["close"].pct_change().rolling(14).std() * 100).fillna(1)
    return out


# ---------------- Feature matrix lengkap ----------------

FEATURE_GROUPS = {
    "indicator": ["rsi", "macd_hist", "ema_trend", "bb_pos", "stoch_k", "atr_pct"],
    "candle": ["engulfing_bull", "engulfing_bear", "hammer", "shooting_star",
               "doji", "marubozu_bull", "marubozu_bear", "three_white", "three_black", "inside_bar"],
    "volume": ["vol_sma_ratio", "obv_slope", "vwap_dev", "volatility_pct"],
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan semua grup fitur dalam satu DataFrame berindeks sama dengan df."""
    f = pd.DataFrame(index=df.index)
    close = df["close"]

    f["rsi"] = rsi(close) / 50 - 1                      # -1..1 ter-normalisasi
    line, sig, hist = macd(close)
    f["macd_hist"] = (hist / close).replace([np.inf, -np.inf], 0).clip(-0.05, 0.05) * 20
    f["ema_trend"] = ((ema(close, 12) - ema(close, 50)) / close).clip(-0.05, 0.05) * 20
    mid, ub, lb = bollinger(close)
    f["bb_pos"] = ((close - lb) / (ub - lb).replace(0, np.nan) * 2 - 1).clip(-2, 2).fillna(0)
    f["stoch_k"] = (stochastic(df) - 50) / 50
    f["atr_pct"] = (atr(df) / close * 100).clip(0, 10) / 5 - 1

    cp = candle_patterns(df)
    for col in FEATURE_GROUPS["candle"]:
        f[col] = cp[col]

    vf = volume_features(df)
    for col in FEATURE_GROUPS["volume"]:
        f[col] = vf[col]

    return f.fillna(0)
