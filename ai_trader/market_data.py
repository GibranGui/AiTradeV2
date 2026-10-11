"""Akses data pasar asli Indodax (public API, tanpa kunci API).

Format pair resmi Indodax memakai garis bawah: btc_idr, eth_idr, sol_idr, ...
(Nama populer seperti "IDRBTC"/"BTC-IDR" otomatis dinormalisasi.)

Endpoint yang dipakai:
  - ticker : https://indodax.com/api/ticker/{pair}
  - trades : https://indodax.com/api/trades/{pair}?count=N
  - candles: tidak ada endpoint candle publik resmi pada API v1, jadi OHLCV
    dibangun sendiri dari agregasi tick transaksi asli (satu sumber cadangan
    v2 tetap dicoba lebih dulu).

Semua fungsi punya fallback ke data sintetis (mode offline) supaya sistem
tetap bisa berjalan & diuji tanpa internet.
"""

from __future__ import annotations

import re
import time
import random
import numpy as np
import pandas as pd
import requests

BASE = "https://indodax.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (AI-Trader-Paper)"}
TIMEOUT = 10

# Aset yang berpasangan dengan IDR di Indodax (untuk normalisasi nama pair)
_KNOWN_ASSETS = {
    "btc", "eth", "sol", "xrp", "ada", "doge", "dot", "avax", "ltc", "bch",
    "trx", "link", "matic", "pol", "shib", "pepe", "uni", "atom", "near",
    "apt", "arb", "op", "inj", "sei", "tia", "ton", "wif", "bonk", "fetch",
    "fet", "render", "rndr", "galaxia", "gala", "sui", "layer", "zcash",
    "zus", "kas", "tao", "ena", "ondo", "jup", "pyth", "strk", "zro", "xlm",
    "usdt", "usdc",
}


def normalize_pair(pair: str) -> str:
    """'IDRBTC'/'BTC-IDR'/'idrbtc' -> 'btc_idr'; 'XRP' -> 'xrp_idr'."""
    p = re.sub(r"[^a-z0-9]", "", pair.lower())
    if not p:
        return "btc_idr"
    # sudah mengandung bentuk <aset>_idr / idr_<aset>?
    if p.endswith("idr"):
        base = p[:-3]
        return f"{base}_idr" if base in _KNOWN_ASSETS or len(base) >= 2 else "btc_idr"
    if p.startswith("idr"):
        base = p[3:]
        return f"{base}_idr" if base in _KNOWN_ASSETS or len(base) >= 2 else "btc_idr"
    if "_" in pair.lower():
        return pair.lower()
    return f"{p}_idr"


def get_ticker(pair: str = "btc_idr") -> dict | None:
    """Harga terakhir, high/low/vol 24 jam (format asli: satukan satuan penuh)."""
    try:
        r = requests.get(f"{BASE}/ticker/{normalize_pair(pair)}",
                         headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        if "error" in d:
            return None
        t = d.get("ticker", d)
        vol = t.get("vol_idr") or t.get("volume") or t.get("vol_btc") or 0
        return {
            "last_price": float(t["last"]),
            "buy": float(t.get("buy", t["last"])),
            "sell": float(t.get("sell", t["last"])),
            "high": float(t.get("high", 0)),
            "low": float(t.get("low", 0)),
            "volume": float(vol),
        }
    except Exception:
        return None


def get_recent_trades(pair: str = "btc_idr", count: int = 500) -> list[dict] | None:
    """Daftar transaksi terbaru ASLI (mendukung analisis volume/flow & candle).

    Catatan: endpoint /trades publik dapat dinonaktifkan oleh Indodax; jika
    gagal, kita bangun 'tick' dari selisih ticker yang di-poll.
    """
    p = normalize_pair(pair)
    try:
        r = requests.get(f"{BASE}/trades/{p}",
                         params={"count": min(int(count), 1000)},
                         headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        if isinstance(d, dict) and "error" in d:
            raise ValueError(d["error"])
        trades = d.get("trades", d if isinstance(d, list) else [])[:count]
        out = []
        for t in trades:
            out.append({
                "price": float(t["price"]),
                "amount": float(t["amount"]),
                "side": t.get("type", t.get("taker_side", "buy")).lower(),
                "time": int(t.get("date_time", t.get("timestamp", 0))),
            })
        return out or None
    except Exception:
        pass
    # fallback: kumpulkan tick live dari ticker selama beberapa detik
    ticks = []
    last_px = None
    t0 = time.time()
    while time.time() - t0 < max(3.0, min(count / 20, 15)):
        tk = get_ticker(p)
        if tk:
            px = tk["last_price"]
            if last_px is None or px != last_px:
                ticks.append({"price": px, "amount": 0.0,
                              "side": "buy" if (last_px and px > last_px) else "sell",
                              "time": int(time.time())})
                last_px = px
        time.sleep(0.5)
    return ticks or None


def build_ohlcv_from_trades(trades: list[dict], period_min: int) -> pd.DataFrame | None:
    """Agregasi tick transaksi asli menjadi candle OHLCV per `period_min` menit."""
    if not trades or len(trades) < 20:
        return None
    df = pd.DataFrame(trades)
    df = df[df["time"] > 0].copy()
    if df.empty:
        return None
    bucket = period_min * 60
    df["ts"] = (df["time"] // bucket) * bucket
    g = df.groupby("ts").agg(
        open=("price", "first"), high=("price", "max"), low=("price", "min"),
        close=("price", "last"), volume=("amount", "sum"),
    ).reset_index()
    if len(g) < 30:
        return None
    g["time"] = pd.to_datetime(g["ts"], unit="s")
    return g[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)


def get_ohlcv(pair: str = "btc_idr", period: str = "60", limit: int = 500) -> pd.DataFrame | None:
    """Candle OHLCV: coba endpoint v2 bila tersedia, jika tidak agregasi tick asli."""
    p = normalize_pair(pair)
    # 1) percobaan endpoint v2 (beberapa wilayah/versi menyediakan)
    try:
        r = requests.get(
            f"{BASE}/public/v2/pairs/{p}/ohlcv",
            params={"period": period, "limit": limit},
            headers=HEADERS, timeout=TIMEOUT,
        )
        if r.ok:
            data = r.json()
            rows = data if isinstance(data, list) else data.get("candles", data.get("data", []))
            if rows:
                df = _parse_ohlcv_rows(rows)
                if df is not None and len(df) >= 30:
                    return df
    except Exception:
        pass
    # 2) bangun candle dari riwayat transaksi asli (API v1 /trades)
    try:
        period_min = max(1, int(float(period)))
    except (TypeError, ValueError):
        period_min = 60
    # bangun candle dari riwayat transaksi asli (API v1 /trades)
    try:
        period_min = max(1, int(float(period)))
    except (TypeError, ValueError):
        period_min = 60
    span_sec = limit * period_min * 60          # rentang waktu yang dibutuhkan
    trades = get_recent_trades(p, count=1000)
    if not trades:
        return None
    all_tr = list(trades)
    oldest = min(t["time"] for t in trades)
    newest = max(t["time"] for t in trades)
    pages_needed = int(np.ceil(span_sec / max(1, newest - oldest))) + 2
    for _ in range(min(pages_needed, 8)):       # hemat: maksimal 8 halaman request
        if min(t["time"] for t in all_tr) < time.time() - span_sec:
            break
        try:
            r = requests.get(f"{BASE}/trades/{p}",
                             params={"count": 1000, "since": oldest},
                             headers=HEADERS, timeout=TIMEOUT)
            if not r.ok:
                break
            page_raw = r.json().get("trades", [])
            page = [
                {"price": float(t["price"]), "amount": float(t["amount"]),
                 "side": t.get("type", "buy").lower(),
                 "time": int(t.get("date_time", 0))} for t in page_raw
            ]
            if not page:
                break
            new_oldest = min(t["time"] for t in page)
            fresh = [t for t in page if t["time"] < oldest]
            if not fresh or new_oldest >= oldest:
                break
            all_tr.extend(fresh)
            oldest = new_oldest
        except Exception:
            break
    df = build_ohlcv_from_trades(all_tr, period_min)
    if df is not None:
        return df.tail(limit).reset_index(drop=True)
    return None


def _parse_ohlcv_rows(rows: list) -> pd.DataFrame | None:
    try:
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
