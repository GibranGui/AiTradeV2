"""Akses data pasar asli Indodax (public API, tanpa kunci API).

Indodax memiliki dua versi API publik:
  - v1: https://indodax.com/api/ticker/{pair}        dan /trade/{pair}
  - public: https://indodax.com/api/public/v2/pairs/{pair}/ohlcv?period=...&limit=...

Semua fungsi punya fallback ke data sintetis (mode offline) supaya sistem
tetap bisa berjalan & diuji tanpa internet.
"""

from __future__ import annotations

import time
import random
import numpy as np
import pandas as pd
import requests

BASE = "https://indodax.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (AI-Trader-Paper)"}
TIMEOUT = 10


def get_ticker(pair: str = "IDRBTC") -> dict | None:
    """Harga terakhir, high/low/vol 24 jam."""
    try:
        r = requests.get(f"{BASE}/ticker/{pair.lower()}", headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        return {
            "last_price": float(d["last"]),
            "high": float(d.get("high", 0)),
            "low": float(d.get("low", 0)),
            "volume": float(d.get("volume", 0)),
        }
    except Exception:
        return None


def get_recent_trades(pair: str = "IDRBTC", count: int = 50) -> list[dict] | None:
    """Daftar transaksi terbaru (mendukung analisis volume/flow)."""
    try:
        r = requests.get(f"{BASE}/trade/{pair.lower()}", headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        trades = r.json().get("trades", [])[:count]
        return [
            {
                "price": float(t["price"]),
                "amount": float(t["amount"]),
                "side": t.get("taker_side", "buy"),
                "time": int(t.get("date_time", 0)),
            }
            for t in trades
        ]
    except Exception:
        return None


def get_ohlcv(pair: str = "IDRBTC", period: str = "60", limit: int = 500) -> pd.DataFrame | None:
    """Candle OHLCV dari endpoint publik v2. period dalam menit (60 = 1 jam)."""
    try:
        r = requests.get(
            f"{BASE}/public/v2/pairs/{pair.lower()}/ohlcv",
            params={"period": period, "limit": limit},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        rows = data if isinstance(data, list) else data.get("candles", data.get("data", []))
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # Normalisasi nama kolom (format bisa berubah)
        colmap = {}
        for c in df.columns:
            lc = str(c).lower()
            if "time" in lc or "date" in lc:
                colmap[c] = "time"
            elif lc == "open":
                colmap[c] = "open"
            elif lc == "high":
                colmap[c] = "high"
            elif lc == "low":
                colmap[c] = "low"
            elif lc == "close":
                colmap[c] = "close"
            elif "vol" in lc:
                colmap[c] = "volume"
        df = df.rename(columns=colmap)
        need = {"time", "open", "high", "low", "close", "volume"}
        if not need.issubset(df.columns):
            return None
        df = df[list(need)].astype(float)
        df["time"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
        return df.dropna().reset_index(drop=True)
    except Exception:
        return None


def synthetic_ohlcv(n: int = 500, seed: int | None = None) -> pd.DataFrame:
    """Data candle sintetis untuk mode offline/backtest tanpa internet."""
    rng = np.random.default_rng(seed or random.randint(0, 10**6))
    price = 1_000_000.0
    closes, opens, highs, lows, vols, times = [], [], [], [], [], []
    t = time.time() - n * 3600
    for _ in range(n):
        drift = rng.normal(0, 0.004)
        o = price
        c = o * (1 + drift)
        h = max(o, c) * (1 + abs(rng.normal(0, 0.0015)))
        l = min(o, c) * (1 - abs(rng.normal(0, 0.0015)))
        v = abs(rng.normal(50, 20)) * (1 + 4 * abs(drift) / 0.004)  # volume naik saat gerak besar
        opens.append(o); closes.append(c); highs.append(h); lows.append(l)
        vols.append(v); times.append(t)
        price = c
        t += 3600
    return pd.DataFrame({
        "time": pd.to_datetime(times, unit="s"),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": vols,
    })


def fetch_ohlcv_or_synthetic(pair: str, period: str = "60", limit: int = 500) -> tuple[pd.DataFrame, bool]:
    """Kembalikan (df, is_live). is_live True jika data asli dari Indodax."""
    df = get_ohlcv(pair, period, limit)
    if df is not None and len(df) >= 50:
        return df, True
    return synthetic_ohlcv(limit), False
