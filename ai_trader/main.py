"""AI Trader — bot trading dengan data pasar asli Indodax (paper trading).

Cara kerja tiap siklus:
 1. Ambil candle OHLCV real dari Indodax -> hitung fitur (indikator, pola
    candle, volume).
 2. Hitung sinyal gabungan semua strategi (bobot dipelajari AI).
 3. Eksekusi paper trade pada harga ticker live (fee 0,3% ikut disimulasikan).
 4. Saat posisi ditutup (TP/SL/sinyal) -> AI belajar dari kesalahan:
    bobot strategi di-adjust & aturan risiko di-tune (brain.json).
 5. Berkala, AI melakukan evolusi mandiri: mutate portofolio, backtest,
    simpan varian terbaik ke strategies.json (self-modifying configuration).

CLI:
  python -m ai_trader.main run      --pair IDRBTC --interval 60
  python -m ai_trader.main evolve   --pair IDRBTC            # belajar offline
  python -m ai_trader.main status                            # cek dompet & otak
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
from ai_trader.paper_broker import PaperBroker, FEE
from ai_trader.learning import (
    load_brain, save_brain, learn_from_trade, evolve, run_backtest, log_experience,
)


def latest_feature_dict(features, i: int) -> dict:
    row = features.iloc[i]
    return {k: float(v) for k, v in row.items()}


def print_banner(pair, is_live):
    src = "INDODAX LIVE (data asli)" if is_live else "OFFLINE (data sintetis)"
    print(f"\n{'=' * 62}\n  AI TRADER — Paper Trading | Pair: {pair} | Sumber: {src}\n{'=' * 62}")


def run_loop(pair: str, interval_min: int, poll_sec: int, evolve_every: int):
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
            if broker.position:
                chg = price / broker.position.entry_price - 1
                pos_txt = f" | POSISI entry={broker.position.entry_price:,.0f} chg={chg:+.2%}"
            print(f"[{ts}] px={price:,.0f} sinyal={sig:+.3f} kas=Rp{broker.cash_idr:,.0f}{pos_txt}")

            # ---- manajemen posisi ----
            if broker.position:
                chg = price / broker.position.entry_price - 1
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
            else:
                if sig > brain["entry_threshold"]:
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
    ap = argparse.ArgumentParser(description="AI Trader belajar-mandiri + Indodax paper trading")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "evolve", "status"])
    ap.add_argument("--pair", default="IDRBTC")
    ap.add_argument("--interval", type=int, default=60, help="candle period (menit)")
    ap.add_argument("--poll", type=int, default=30, help="detik antar-siklus")
    ap.add_argument("--evolve-every", type=int, default=30, help="menit antar-evolusi")
    a = ap.parse_args()

    if a.cmd == "run":
        run_loop(a.pair, a.interval, a.poll, a.evolve_every)
    elif a.cmd == "evolve":
        do_evolve(a.pair, a.interval)
    elif a.cmd == "status":
        do_status()


if __name__ == "__main__":
    main()
