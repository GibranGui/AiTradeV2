"""Altcoin Scanner — memindai SELURUH pasar IDR untuk menemukan peluang tren naik.

Alih-alih hanya memantau satu koin, tiap siklus bot:
 1. Mengambil 24 jam ticker SEMUA pasar berpasangan IDR sekaligus (1 request).
    Sumber utama: mirror resmi Binance-IDR (mesin likuiditas yang sama dengan
    Indodax sejak akuisisi 2022; API publik Indodax tidak menyediakan endpoint
    daftar-pasar). Harga eksekusi tetap memakai ticker live Indodax.
 2. Menyaring: minimal likuiditas (volume 24 jam), bukan stablecoin.
 3. Preranking cepat berbasis harga (momentum + volume) -> ambil kandidat teratas.
 4. Analisis mendalam per kandidat: candle OHLCV asli -> indikator, pola
    candlestick, volume -> sinyal gabungan strategi AI (bobot terpelajari).
 5. Skor akhir = rata-rata momentum harga & konfluensi sinyal teknikal.
    Kandidat dengan skor >= ambang dianggap "peluang tren naik" dan bisa dibeli.

Pola desain mengikuti pendekatan pipeline gstack (sweep -> filter -> rank ->
act) yang dikhususkan untuk pemindaian pasar kripto.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .market_data import fetch_ohlcv_or_synthetic, normalize_pair
from .features import build_features
from .strategies.portfolio import combined_signal, normalize_signal_features

BINANCE_VISION = "https://data-api.binance.vision"
BINANCE_HOSTS = ("https://data-api.binance.info", BINANCE_VISION,
                 "https://api.binance.com")
HEADERS = {"User-Agent": "Mozilla/5.0 (AI-Trader-Scanner)"}
TIMEOUT = 15

# ---- satu sumber kebenaran untuk ambang keputusan (user minta threshold   ----
# ---- dilonggarkan; nilai lama: min_score=0.62, floor sinyal=0.15).        ----
# Perubahan ini HANYA melonggarkan gerbang frekuensi — logika skor, filter
# risiko, fee, SL/TP, dan mekanisme belajar tidak disentuh sama sekali.
DEFAULT_MIN_SCORE = 0.55   # pelonggaran moderat dari 0.62 (bukti funnel CI:
                           #   kandidat terbaik mentok di 0.45-0.57 saat pasar
                           #   sideways; 0.55 mulai membuka pintu tanpa banjir)
SIGNAL_FLOOR = 0.10        # konfirmasi arah minimal; diturunkan dari 0.15
                           #   SEJALAN pelonggaran skor agar gerbang tetap
                           #   konsisten scanner <-> main.py

# stablecoin / aset non-spekulatif — jangan pernah diperdagangkan
STABLECOINS = {
    "usdt", "usdc", "busd", "dai", "fdusd", "tusd", "usdp", "idrxx", "idrtb",
    "idr", "xaut", "paxg", "tether",
}


def _is_stable_symbol(sym: str) -> bool:
    """Tangkap bentuk BUSDBIDR/USDTBIDR (aset turunan stablecoin ber-suffix B)."""
    for st in ("USDT", "USDC", "BUSD", "DAI", "FDUSD", "TUSD", "USDP"):
        if sym == st + "IDR" or sym == st + "BIDR":
            return True
    return False


def _binance_universe(min_vol_idr: float) -> list[dict]:
    """Satu request: semua ticker 24 jam pasar IDR di Binance-IDR."""
    for host in BINANCE_HOSTS:
        try:
            r = requests.get(f"{host}/api/v3/ticker/24hr",
                             headers=HEADERS, timeout=TIMEOUT)
            if not r.ok:
                continue
            rows = r.json()
            out = []
            for t in rows:
                sym = t["symbol"]
                if not (sym.endswith("IDR") or sym.endswith("BIDR")):
                    continue
                # BIDR = pasangan stablecoin rupiah Indodax (mis. SOLBIDR);
                # normalize ke base_idr agar sama dengan format pair Indodax.
                if sym.endswith("BIDR"):
                    base = sym[:-4].lower()
                else:
                    base = sym[:-3].lower()
                if len(base) < 2 or _is_stable_symbol(sym):
                    continue
                # stablecoin & turunannya: usdt, usdtb, busd, busdb, dst.
                if base.rstrip("b") in STABLECOINS:
                    continue
                qv = float(t.get("quoteVolume", 0))
                if qv < min_vol_idr:
                    continue
                out.append({
                    "pair": f"{base}_idr",
                    "symbol": base.upper() + "IDR",
                    "last": float(t["lastPrice"]),
                    "chg24h_pct": float(t.get("priceChangePercent", 0)),
                    "vol_idr_24h": qv,
                    "high24h": float(t.get("highPrice", 0)),
                    "low24h": float(t.get("lowPrice", 0)),
                })
            if out:
                # deduplikasi jika base yang sama muncul dua kali:
                # pilih pasar dengan volume terbesar sebagai representasi.
                best: dict[str, dict] = {}
                for u in out:
                    cur = best.get(u["pair"])
                    if cur is None or u["vol_idr_24h"] > cur["vol_idr_24h"]:
                        best[u["pair"]] = u
                return list(best.values())
        except Exception:
            continue
    return []


def _indodax_universe(min_vol_idr: float) -> list[dict]:
    """Fallback: keranjang (basket) altcoin populer, cek ticker live Indodax."""
    basket = [
        "btc", "eth", "sol", "xrp", "ada", "doge", "dot", "avax", "ltc", "trx",
        "link", "matic", "shib", "pepe", "uni", "atom", "near", "apt", "arb",
        "op", "inj", "sei", "tia", "ton", "wif", "bonk", "fetch", "render",
        "gala", "sui", "kas", "tao", "ena", "ondo", "jup", "pyth", "strk",
        "xlm", "bch", "algo", "sand", "mana", "aave", "fil", "icp", "ar",
        "galxe", "ray", "dydx", "loom", "fio", "ocean", "glmr", "knc",
    ]
    out = []
    for a in basket:
        p = f"{a}_idr"
        try:
            r = requests.get(f"https://indodax.com/api/ticker/{p}",
                             headers=HEADERS, timeout=8)
            if not r.ok:
                continue
            d = r.json()
            if "error" in d:
                continue
            t = d.get("ticker", d)
            vol = float(t.get("vol_idr") or 0)
            if vol < min_vol_idr:
                continue
            last = float(t["last"])
            high, low = float(t.get("high", last)), float(t.get("low", last))
            out.append({
                "pair": p, "symbol": p.split("_")[0].upper() + "IDR",
                "last": last,
                "chg24h_pct": (last / ((high + low) / 2) - 1) * 100,
                "vol_idr_24h": vol, "high24h": high, "low24h": low,
            })
        except Exception:
            continue
    return out


def get_universe(min_vol_idr: float = 500_000_000) -> tuple[list[dict], str]:
    """Semua koin berpasangan IDR yang layak dipantau. Return (daftar, sumber)."""
    uni = _binance_universe(min_vol_idr)
    if len(uni) >= 5:
        return uni, "binance-idr"
    uni2 = _indodax_universe(min_vol_idr)
    return (uni2, "indodax-basket") if uni2 else ([], "offline")


def prescreen(universe: list[dict], top_n: int,
              max_chg_pct: float = 25.0) -> list[dict]:
    """Preranking cepat tanpa data historis: momentum sehat + likuiditas.

    - chg24h > 0 (tren naik), tapi tidak > max_chg_pct (hindari pump & dump).
    - posisi harga dalam range 24 jam menunjukkan kekuatan beli.
    - skor digabung dengan akar volume (likuiditas penting untuk exit).
    """
    scored = []
    for u in universe:
        chg = u["chg24h_pct"]
        if chg <= 0 or chg > max_chg_pct:
            continue
        rng = u["high24h"] - u["low24h"]
        pos_in_range = (u["last"] - u["low24h"]) / rng if rng > 0 else 0.5
        liq = np.log10(max(u["vol_idr_24h"], 1)) / 11.0      # ~0..1
        score = 0.45 * min(chg / max_chg_pct, 1.0) + 0.35 * pos_in_range + 0.20 * liq
        scored.append((score, u))
    scored.sort(key=lambda x: -x[0])
    return [u for _, u in scored[:top_n]]


def _passes_deep_filter(r: dict, max_chg_pct: float) -> bool:
    """Koin layak dianalisis mendalam: sedang naik ATAU baru mulai bangkit
    (harga dekat high 24 jam meski chg24h masih negatif — pola awal tren)."""
    chg = r["chg24h_pct"]
    if 0 < chg <= max_chg_pct:
        return True
    rng = r["high24h"] - r["low24h"]
    near_high = rng > 0 and (r["last"] - r["low24h"]) / rng >= 0.8
    return bool(near_high and -3.0 <= chg <= 0)


class CoinScanner:
    """Memindai kandidat & menghitung skor peluang tren naik per koin."""

    def __init__(self, strategies, brain, interval_min: int = 60,
                 limit: int = 300, cache_sec: int = 240):
        self.strategies = strategies
        self.brain = brain
        self.interval_min = interval_min
        self.limit = limit
        self.cache_sec = cache_sec
        self._cache: dict[str, tuple[float, dict]] = {}   # pair -> (ts, hasil)

    def analyze(self, coin: dict, diag=None) -> dict | None:
        pair = coin["pair"]
        hit = self._cache.get(pair)
        if hit and time.time() - hit[0] < self.cache_sec:
            return hit[1]
        try:
            df, is_live = fetch_ohlcv_or_synthetic(
                pair, period=str(self.interval_min), limit=self.limit)
        except Exception as e:                       # jangan hentikan seluruh siklus
            if diag:
                diag.data_bad(coin["symbol"], "fetch_exception", str(e)[:80])
            return None
        if df is None or len(df) < 60 or not is_live:
            # JANGAN pernah trading/skor dari data sintetis — hanya candle asli.
            if diag:
                nbar = 0 if df is None else len(df)
                why = ("sintetis" if (df is not None and nbar >= 60 and not is_live)
                       else f"bar={nbar}<60")
                diag.data_bad(coin["symbol"], "candle_data_incomplete", why)
            return None
        feats = normalize_signal_features(build_features(df))
        i = len(feats) - 1
        sig = combined_signal(self.strategies, feats, i)

        close = feats["close"]
        n = len(close)
        # momentum multi-horizon pada candle asli
        def mom(bars):
            j = max(0, n - bars)
            return float(close.iloc[-1] / max(1e-9, close.iloc[j]) - 1)
        m5, m20 = mom(5), mom(20)
        ema_fast_ok = float(feats["ema_trend"][i]) > 0       # tren EMA positif
        # volume mendukung (OBV slope & rasio volume terhadap SMA)
        vol_sup = float(np.clip(float(feats["obv_slope"][i]) * 0.5
                                + float(feats["vol_sma_ratio"][i]) * 0.25, -1, 1))
        uptrend = 1.0 if (m20 > 0 and ema_fast_ok) else 0.0

        # skor konfluensi teknikal [-1..1] -> [0..1]
        tech = (sig + 1) / 2
        trend_score = float(np.clip(
            0.35 * tech + 0.25 * np.tanh(m5 * 20) + 0.20 * uptrend
            + 0.10 * np.tanh(m20 * 10) + 0.10 * max(0.0, vol_sup), 0, 1))

        result = {
            "pair": pair, "symbol": coin["symbol"], "score": trend_score,
            "signal": sig, "mom5": m5, "mom20": m20, "uptrend": bool(uptrend),
            "vol_support": vol_sup, "last_close": float(close.iloc[-1]),
            "live": is_live, "chg24h_pct": coin["chg24h_pct"],
            "vol_idr_24h": coin["vol_idr_24h"],
            "features": {k: float(v) for k, v in feats.iloc[i].items()},
        }
        self._cache[pair] = (time.time(), result)
        return result

    def scan(self, universe: list[dict], deep_n: int = 8, diag=None) -> list[dict]:
        """Sweep -> prescreen -> analisis mendalam -> ranking final.

        Prinsip gstack "sweep wide, act narrow": kandidat yang layak dianalisis
        (tren naik / baru bangkit + likuid) diurutkan berdasarkan likuiditas,
        lalu SEMUA-nya dianalisis mendalam (bukan hanya segelintir), dibatasi
        `deep_n` sebagai pagar biaya/rate-limit — bukan sebagai batas ketat
        ketika peluang tersebar di banyak koin.
        """
        cands = [u for u in universe if _passes_deep_filter(u, max_chg_pct=25.0)]
        # fallback: jika tidak ada yang lolos filter ketat, ambil top momentum
        if not cands:
            cands = prescreen(universe, top_n=min(deep_n, len(universe)))
        # urutkan prioritas: volume terbesar dulu (likuiditas = mudah exit)
        cands.sort(key=lambda u: -u["vol_idr_24h"])
        cands = cands[:max(deep_n, 10)]          # minimal 10 kandidat per siklus
        if diag:
            diag.deep_start(len(cands))
        results = []
        for c in cands:
            r = self.analyze(c, diag=diag)
            if r:
                if diag:
                    diag.data_ok()
                results.append(r)
            time.sleep(0.1)  # hemat rate-limit
        results.sort(key=lambda x: -x["score"])
        return results


def opportunities(results: list[dict], min_score: float,
                  require_confluence: bool = True, diag=None,
                  signal_floor: float | None = None) -> list[dict]:
    """Filter hasil scan menjadi daftar 'peluang tren naik'.

    BUG FIX (akar masalah 0 transaksi): skor peluang dihitung dengan rumus
    ADITIF — 35% konfluensi strategi + 25% momentum 5-bar + 20% uptrend + ...
    Artinya koin dengan momentum/volume kuat tapi sinyal strategi masih kecil
    (mis. 0.19) bisa menembus ambang 0.62, padahal konfirmasi arah WAJIB
    (main.py menolak bila signal < ~0.15). Sebelum fix ini, setiap kandidat
    yang lolos gerbang skor justru GUGUR di gerbang konfirmasi — tanpa jejak
    log apa pun, sehingga bot tampak "hanya scan". Sekarang kedua gerbang
    disatumkan di sini: kandidat harus memenuhi SKOR dan SINYAL sekaligus,
    dan setiap penolakan dicatat lewat diagnostic logging.

    REDESAIN (filosofi baru user): `min_score` & `signal_floor` kini adalah
    ambang adaptif MILIK AI (dari brain/strategies.json), bukan konstanta
    template. Nilai default hanya fallback bila pemanggil tidak mengirim.
    """
    if signal_floor is None:
        signal_floor = SIGNAL_FLOOR
    out = []
    for r in results:
        sym = r["symbol"]
        if r["score"] < min_score:
            if diag:
                diag.reject(sym, "score_filter", "score_below_threshold",
                            actual=f"{r['score']:.3f}", required=f"{min_score:.2f}")
            continue
        if not r["uptrend"]:
            if diag:
                diag.reject(sym, "trend_filter", "not_uptrend",
                            actual=f"mom20={r['mom20']:+.2%} ema_trend<=0")
            continue
        if require_confluence:
            thr = signal_floor              # ambang adaptif AI; selaras dgn
                                            # plafon entry di main.py
            sig = r["signal"]
            m5 = r["mom5"]
            if sig < thr:
                if diag:
                    diag.reject(sym, "strategy_confirmation",
                                "signal_below_entry_floor",
                                actual=f"signal={sig:+.3f}", required=f">={thr}")
                continue
            # konfluensi momentum: sinyal beli kuat tapi 5 bar terakhir turun
            # tajam = pola reversal sedang terjadi — jangan masuk.
            if sig > signal_floor and m5 < -0.005:
                if diag:
                    diag.reject(sym, "momentum_conflict",
                                "bullish_signal_vs_negative_m5",
                                actual=f"signal={sig:+.3f} mom5={m5:+.3%}",
                                required="mom5>=-0.5%")
                continue
        out.append(r)
    return out
