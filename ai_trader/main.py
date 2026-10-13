"""AI Trader — bot trading MULTI-KOIN dengan data pasar asli (paper trading).

Bot tidak lagi terpaku pada satu koin: tiap siklus ia MEMINDAI seluruh pasar
berpasangan IDR (via mirror Binance-IDR + ticker live Indodax), lalu hanya
berdagang di koin yang menunjukkan PELUANG TREN NAIK.

Cara kerja tiap siklus (mode scan):
 1. Sweep: ambil 24 jam ticker semua pasar IDR -> filter stablecoin & likuiditas.
 2. Prescreen: ranking cepat momentum harga -> kandidat teratas (default 8).
 3. Analisis mendalam per kandidat: candle OHLCV asli -> indikator, pola
    candlestick, volume -> sinyal gabungan strategi AI (bobot terpelajari).
 4. Kelola posisi SEMUA koin yang dipegang: TP/SL/trailing per posisi,
    exit saat sinyal berbalik.
 5. Entry: beli koin skor tertinggi yang belum dipegang (maks 5 posisi,
    alokasi per-trade = alloc_frac x kas tersisa).
 6. Saat posisi ditutup -> AI belajar dari kesalahan (credit assignment ke
    bobot strategi & aturan risiko, disimpan di brain.json).
 7. Berkala, AI berevolusi mandiri: mutate portofolio strategi, backtest,
    simpan varian terbaik (self-modifying configuration, strategies.json).

CLI:
  python -m ai_trader.main run      --pair IDRBTC --interval 60   # satu koin (lama)
  python -m ai_trader.main scan     --interval 60                 # MULTI-KOIN (baru)
  python -m ai_trader.main list                                   # daftar peluang tren naik sekarang
  python -m ai_trader.main evolve   --pair IDRBTC                 # belajar offline
  python -m ai_trader.main status                                 # cek dompet & otak
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# pastikan bisa dijalankan sebagai script langsung maupun modul
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_trader.market_data import fetch_ohlcv_or_synthetic, get_ticker
from ai_trader.features import build_features
from ai_trader.strategies.portfolio import normalize_signal_features
from ai_trader.strategies.portfolio import (
    Strategy, combined_signal, load_portfolio, save_portfolio,
)
from ai_trader.paper_broker import PaperBroker, FEE, MAX_POSITIONS
from ai_trader.scanner import CoinScanner, get_universe, opportunities
from ai_trader.learning import (
    load_brain, save_brain, learn_from_trade, evolve, run_backtest, log_experience,
)


def latest_feature_dict(features, i: int) -> dict:
    row = features.iloc[i]
    return {k: float(v) for k, v in row.items()}


def print_banner(pair, is_live):
    src = "INDODAX LIVE (data asli)" if is_live else "OFFLINE (data sintetis)"
    print(f"\n{'=' * 62}\n  AI TRADER — Paper Trading | Pair: {pair} | Sumber: {src}\n{'=' * 62}")


def run_loop(pair: str, interval_min: int, poll_sec: int, evolve_every: int,
             scan_mode: bool = False, deep_n: int = 8, min_score: float = 0.62,
             min_vol_idr: float = 500_000_000):
    if scan_mode:
        return run_scan_loop(interval_min, poll_sec, evolve_every,
                             deep_n, min_score, min_vol_idr)
    broker = PaperBroker()
    brain = load_brain()
    strategies = load_portfolio()

    df, is_live = fetch_ohlcv_or_synthetic(pair, period=str(interval_min), limit=400)
    features = normalize_signal_features(build_features(df))
    print_banner(pair, is_live)
    print(f"Kas awal: Rp {broker.cash_idr:,.0f} | Strategi aktif: {len(strategies)}")
    print(f"Brain: thr={brain['entry_threshold']:.2f} SL={brain['stop_loss_pct']:.2%} "
          f"TP={brain['take_profit_pct']:.2%} alloc={brain['alloc_frac']:.0%}\n")

    last_evolve = time.time()
    cycle = 0
    try:
        while True:
            cycle += 1
            # refresh data pasar secara berkala
            if cycle % 10 == 1 or not is_live:
                df, is_live = fetch_ohlcv_or_synthetic(pair, period=str(interval_min), limit=400)
                features = normalize_signal_features(build_features(df))

            price = broker.current_price(pair)
            if price is None and len(df):
                price = float(df["close"].iloc[-1])
            if price is None:
                time.sleep(poll_sec)
                continue

            i = len(features) - 1
            sig = combined_signal(strategies, features, i)
            ts = time.strftime("%H:%M:%S")
            pos_txt = ""
            if pair in broker.positions:
                p0 = broker.positions[pair]
                chg = price / p0.entry_price - 1
                pos_txt = f" | POSISI entry={p0.entry_price:,.0f} chg={chg:+.2%}"
            print(f"[{ts}] px={price:,.0f} sinyal={sig:+.3f} kas=Rp{broker.cash_idr:,.0f}{pos_txt}")

            # ---- manajemen posisi pair ini ----
            if pair in broker.positions:
                chg = price / broker.positions[pair].entry_price - 1
                reason = None
                if chg <= -brain["stop_loss_pct"]:
                    reason = "stop_loss"
                elif chg >= brain["take_profit_pct"]:
                    reason = "take_profit"
                elif sig < -brain["entry_threshold"] * 0.5:
                    reason = "signal_exit"
                if reason:
                    trade = broker.sell(pair, price, reason)
                    if trade:
                        lessons = learn_from_trade(trade, strategies, brain)
                        save_brain(brain)
                        save_portfolio(strategies)
                        print(f"  >> JUAL ({reason}) PnL: Rp {trade['pnl']:,.0f} "
                              f"({trade['pnl_pct']:+.2%})")
                        for l in lessons:
                            print(f"     [BELAJAR] {l}")
            elif sig > brain["entry_threshold"]:
                feats_snap = latest_feature_dict(features, i)
                trade = broker.buy(pair, price, feats_snap, alloc=brain["alloc_frac"])
                if trade:
                    print(f"  >> BELI @ {price:,.0f} (sinyal {sig:+.2f} > "
                          f"threshold {brain['entry_threshold']:.2f})")

            # ---- evolusi mandiri berkala ----
            if time.time() - last_evolve > evolve_every * 60:
                print("  >> AI melakukan evolusi strategi (backtest self-improvement)...")
                strategies, res = evolve(strategies, df, features, brain)
                save_portfolio(strategies)
                last_evolve = time.time()
                print(f"     hasil evolusi: return={res['return_pct']:+.2f}% "
                      f"winrate={res['win_rate']:.0%} fitness={res['fitness']:.2f}")

            time.sleep(poll_sec)
    except KeyboardInterrupt:
        print("\nDihentikan. Menyimpan state...")
        save_brain(brain)
        save_portfolio(strategies)
        broker.save()


def do_evolve(pair: str, interval_min: int):
    """Perintah satu-kali: biarkan AI belajar/berevolusi dari data historis."""
    brain = load_brain()
    strategies = load_portfolio()
    df, is_live = fetch_ohlcv_or_synthetic(pair, period=str(interval_min), limit=600)
    features = normalize_signal_features(build_features(df))
    print(f"Evolusi pada {pair} ({'live' if is_live else 'sintetis'}), {len(df)} candle...")
    before = run_backtest(strategies, df, features, brain)
    print(f"  sebelum : return={before['return_pct']:+.2f}% winrate={before['win_rate']:.0%} "
          f"fitness={before['fitness']:.2f}")
    strategies, after = evolve(strategies, df, features, brain)
    print(f"  sesudah : return={after['return_pct']:+.2f}% winrate={after['win_rate']:.0%} "
          f"fitness={after['fitness']:.2f}")
    save_portfolio(strategies)
    print(f"  tersimpan -> {len(strategies)} strategi (strategies.json)")


def run_scan_loop(interval_min: int, poll_sec: int, evolve_every: int,
                  deep_n: int, min_score: float, min_vol_idr: float):
    """MODE MULTI-KOIN: pindai seluruh pasar IDR tiap siklus, tradingi koin
    yang menunjukkan peluang tren naik. Tidak terpaku pada satu koin."""
    broker = PaperBroker()
    brain = load_brain()
    strategies = load_portfolio()
    scanner = CoinScanner(strategies, brain, interval_min=interval_min)

    print(f"\n{'=' * 62}\n"
          f"  AI TRADER MULTI-COIN — paper trading | scan pasar IDR\n"
          f"{'=' * 62}")
    print(f"Kas awal : Rp {broker.cash_idr:,.0f} | posisi maks: {MAX_POSITIONS}")
    print(f"Strategi : {len(strategies)} | ambang skor entry: {min_score:.2f}")
    print(f"Brain    : thr={brain['entry_threshold']:.2f} "
          f"SL={brain['stop_loss_pct']:.2%} TP={brain['take_profit_pct']:.2%} "
          f"alloc={brain['alloc_frac']:.0%}\n")

    last_evolve = time.time()
    cycle = 0
    try:
        while True:
            cycle += 1
            ts = time.strftime("%H:%M:%S")
            universe, src = get_universe(min_vol_idr=min_vol_idr)
            results = scanner.scan(universe, deep_n=deep_n)
            opps = opportunities(results, min_score)

            # harga live Indodax untuk semua pair yang relevan (posisi + kandidat)
            pairs = {p["pair"] for p in results} | set(broker.positions)
            prices: dict[str, float] = {}
            for p in pairs:
                px = broker.current_price(p)
                if px is None:
                    r = next((x for x in results if x["pair"] == p), None)
                    px = r["last_close"] if r else None
                if px:
                    prices[p] = px

            eq = broker.equity(prices)
            pos_txt = ", ".join(
                f"{k.split('_')[0].upper()}({prices[k] / broker.positions[k].entry_price - 1:+.1%})"
                if k in prices else k.split('_')[0].upper()
                for k in broker.positions) or "-"
            print(f"[{ts}] pasar={len(universe)} ({src}) kandidat={len(results)} "
                  f"peluang={len(opps)} ekuitas=Rp{eq:,.0f} posisi=[{pos_txt}]")

            # ---- manajemen SEMUA posisi (TP/SL/trailing/sinyal) ----
            for pair in list(broker.positions):
                pos = broker.positions[pair]
                price = prices.get(pair)
                if price is None:
                    continue
                chg = price / pos.entry_price - 1
                peak = pos.entry_features.get("_peak_chg", 0.0)
                if chg > peak:
                    pos.entry_features["_peak_chg"] = chg
                    peak = chg
                reason = None
                if chg <= -brain["stop_loss_pct"]:
                    reason = "stop_loss"
                elif chg >= brain["take_profit_pct"]:
                    reason = "take_profit"
                elif peak >= brain["take_profit_pct"] * 0.6 and \
                        chg < peak - max(brain["stop_loss_pct"], 0.02):
                    reason = "trailing_stop"      # kunci profit saat tren patah
                else:
                    res = next((x for x in results if x["pair"] == pair), None)
                    sig_now = res["signal"] if res else None
                    if sig_now is None:
                        df, _ = fetch_ohlcv_or_synthetic(
                            pair, period=str(interval_min), limit=300)
                        if df is not None and len(df) >= 60:
                            f2 = normalize_signal_features(build_features(df))
                            sig_now = combined_signal(strategies, f2, len(f2) - 1)
                    if sig_now is not None and sig_now < -brain["entry_threshold"] * 0.5:
                        reason = "signal_exit"
                if reason:
                    trade = broker.sell(pair, price, reason)
                    if trade:
                        lessons = learn_from_trade(trade, strategies, brain)
                        save_brain(brain)
                        save_portfolio(strategies)
                        print(f"  >> JUAL {pair.split('_')[0].upper()} ({reason}) "
                              f"PnL: Rp {trade['pnl']:,.0f} ({trade['pnl_pct']:+.2%})")
                        for l in lessons:
                            print(f"     [BELAJAR] {l}")

            # ---- entry: beli peluang tren naik terbaik (multi-koin) ----
            held = set(broker.positions)
            for opp in opps:                      # sudah terurut skor desc
                if len(broker.positions) >= MAX_POSITIONS:
                    break
                pair = opp["pair"]
                if pair in held:
                    continue
                thr = min(brain["entry_threshold"], 0.15)
                if opp["signal"] < thr:
                    continue      # konfirmasi arah: sinyal gabungan tidak boleh negatif kuat
                price = prices.get(pair) or opp["last_close"]
                feats_snap = {k: v for k, v in opp["features"].items()}
                feats_snap["_scan_score"] = opp["score"]
                trade = broker.buy(pair, price, feats_snap, alloc=brain["alloc_frac"])
                if trade:
                    held.add(pair)
                    print(f"  >> BELI {pair.split('_')[0].upper()} @ {price:,.0f} "
                          f"(skor={opp['score']:.2f} sinyal={opp['signal']:+.2f} "
                          f"mom24h={opp['chg24h_pct']:+.1f}%)")

            # ---- tampilkan papan peringkat peluang ----
            for r in results[:5]:
                mark = "*" if r["pair"] in broker.positions else " "
                print(f"     {mark}{r['symbol']:<12} skor={r['score']:.2f} "
                      f"sig={r['signal']:+.2f} m20={r['mom20']:+.2%} "
                      f"vol24h=Rp{r['vol_idr_24h'] / 1e9:.1f}M "
                      f"{'UPTREND' if r['uptrend'] else ''}")

            # ---- evolusi mandiri berkala (backtest BTC sebagai proksi) ----
            if time.time() - last_evolve > evolve_every * 60:
                print("  >> AI melakukan evolusi strategi (self-improvement)...")
                df, _ = fetch_ohlcv_or_synthetic("btc_idr", period=str(interval_min), limit=500)
                features = normalize_signal_features(build_features(df))
                strategies, res = evolve(strategies, df, features, brain)
                save_portfolio(strategies)
                scanner.strategies = strategies
                last_evolve = time.time()
                print(f"     hasil evolusi: return={res['return_pct']:+.2f}% "
                      f"winrate={res['win_rate']:.0%} fitness={res['fitness']:.2f}")

            time.sleep(poll_sec)
    except KeyboardInterrupt:
        print("\nDihentikan. Menyimpan state...")
        save_brain(brain)
        save_portfolio(strategies)
        broker.save()


def do_list(deep_n: int, min_score: float, min_vol_idr: float, interval_min: int):
    """Satu kali scan: tampilkan daftar koin dengan peluang tren naik."""
    brain = load_brain()
    strategies = load_portfolio()
    scanner = CoinScanner(strategies, brain, interval_min=interval_min)
    universe, src = get_universe(min_vol_idr=min_vol_idr)
    print(f"Memindai {len(universe)} pasar IDR (sumber: {src})...")
    results = scanner.scan(universe, deep_n=deep_n)
    print(f"\n{'KOIN':<12} {'SKOR':>6} {'SINYAL':>8} {'MOM24H':>8} {'MOM20B':>8} "
          f"{'VOL24H(IDR)':>14} TREN")
    print("-" * 70)
    for r in results:
        flag = "NAIK ✓" if r["uptrend"] and r["score"] >= min_score else \
               ("netral" if r["uptrend"] else "turun")
        print(f"{r['symbol']:<12} {r['score']:>6.2f} {r['signal']:>+8.2f} "
              f"{r['chg24h_pct']:>7.1f}% {r['mom20']:>+7.2%} "
              f"{r['vol_idr_24h']:>14,.0f} {flag}")
    opps = opportunities(results, min_score)
    print(f"\n=> {len(opps)} PELUANG TREN NAIK (skor >= {min_score:.2f}): "
          + ", ".join(o["symbol"] for o in opps))


def do_status():
    base = os.path.join(os.path.dirname(__file__), "..", "data")
    for fn in ("paper_wallet.json", "brain.json", "strategies.json"):
        p = os.path.join(base, fn)
        if os.path.exists(p):
            print(f"\n--- {fn} ---")
            print(open(p).read()[:1500])
    exp = os.path.join(base, "experience_buffer.jsonl")
    if os.path.exists(exp):
        lines = open(exp).read().strip().splitlines()
        print(f"\n--- pengalaman terakhir ({len(lines)} entri) ---")
        for l in lines[-3:]:
            print(l[:300])


def main():
    ap = argparse.ArgumentParser(description="AI Trader belajar-mandiri + Indodax paper trading multi-koin")
    ap.add_argument("cmd", nargs="?", default="scan",
                    choices=["run", "scan", "list", "evolve", "status"])
    ap.add_argument("--pair", default="IDRBTC")
    ap.add_argument("--interval", type=int, default=60, help="candle period (menit)")
    ap.add_argument("--poll", type=int, default=30, help="detik antar-siklus")
    ap.add_argument("--evolve-every", type=int, default=30, help="menit antar-evolusi")
    ap.add_argument("--deep", type=int, default=8, help="jumlah kandidat untuk analisis mendalam")
    ap.add_argument("--min-score", type=float, default=0.62, help="ambang skor peluang tren naik")
    ap.add_argument("--min-vol", type=float, default=500_000_000,
                    help="volume 24 jam minimum (IDR) agar koin dilirik")
    a = ap.parse_args()

    if a.cmd == "run":
        run_loop(a.pair, a.interval, a.poll, a.evolve_every,
                 scan_mode=(a.pair.lower() in ("", "all", "scan")),
                 deep_n=a.deep, min_score=a.min_score, min_vol_idr=a.min_vol)
    elif a.cmd == "scan":
        run_scan_loop(a.interval, a.poll, a.evolve_every,
                      a.deep, a.min_score, a.min_vol)
    elif a.cmd == "list":
        do_list(a.deep, a.min_score, a.min_vol, a.interval)
    elif a.cmd == "evolve":
        do_evolve(a.pair, a.interval)
    elif a.cmd == "status":
        do_status()


if __name__ == "__main__":
    main()
