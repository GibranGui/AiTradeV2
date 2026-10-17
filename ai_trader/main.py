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
from ai_trader.scalper import (Scalper, scalp_exit_decision,
                               SCALP_INTERVAL_MIN, SCALP_MAX_POSITIONS)
from ai_trader.scanner import (CoinScanner, get_universe, opportunities,
                             DEFAULT_MIN_SCORE, SIGNAL_FLOOR)
from ai_trader.strategies.portfolio import portfolio_gates
from ai_trader.gates import apply_silence_decay, sync_brain

# Ambang exit-sinyal TETAP dan longgar: keputusan keluar tidak boleh ikut
# threshold entry yang dinaikkan AI (bug "exit tersandera": thr 0.45 membuat
# signal_exit butuh sig<-0.225 = hanya 0,8% bar -> posisi menggantung sampai
# SL penuh; WR trend nyata cuma 7%, EV -Rp26.800/trade).
SIGNAL_EXIT_FLOOR = 0.10
from ai_trader.diag import CycleDiag
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
             scan_mode: bool = False, deep_n: int = 8, min_score: float = DEFAULT_MIN_SCORE,
             min_vol_idr: float = 500_000_000, max_minutes: float = 0.0):
    if scan_mode:
        return run_scan_loop(interval_min, poll_sec, evolve_every,
                             deep_n, min_score, min_vol_idr, max_minutes)
    _deadline = time.time() + (max_minutes * 60 if max_minutes > 0 else 1e12)
    broker = PaperBroker()
    brain = load_brain()
    strategies = load_portfolio()

    df, is_live = fetch_ohlcv_or_synthetic(pair, period=str(interval_min), limit=400)
    features = normalize_signal_features(build_features(df))
    print_banner(pair, is_live)
    print(f"Kas awal: Rp {broker.cash_idr:,.0f} | Strategi aktif: {len(strategies)}")
    print(f"Brain: thr={brain['entry_threshold']:.2f} SL={brain['stop_loss_pct']:.2%} "
          f"TP={brain['take_profit_pct']:.2%} alloc={brain['alloc_frac']:.0%}\n")

    cycle = 0
    try:
        while time.time() < _deadline:
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
                elif sig < -SIGNAL_EXIT_FLOOR:   # exit tidak boleh tersandera threshold entry
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
          f"trades_bt={after.get('trades', 0)} thr_belajar={after.get('entry_thr_used', float('nan')):.2f} "
          f"fitness={after['fitness']:.2f}")
    save_portfolio(strategies)
    save_brain(brain)      # ambang hasil belajar (entry_threshold) ikut tersimpan
    g = portfolio_gates(strategies)
    print(f"  gerbang belajar AI -> entry_thr={g['entry_thr']:.2f} "
          f"score_min={g['score_min']:.2f} floor={g['signal_floor']:.2f} "
          f"scalp={g['scalp_strong']:.2f}/{g['scalp_floor']:.2f}")
    print(f"  tersimpan -> {len(strategies)} strategi (strategies.json)")


def run_scan_loop(interval_min: int, poll_sec: int, evolve_every: int,
                  deep_n: int, min_score: float, min_vol_idr: float,
                  max_minutes: float = 0.0):
    """MODE MULTI-KOIN: pindai seluruh pasar IDR tiap siklus, tradingi koin
    yang menunjukkan peluang tren naik. Tidak terpaku pada satu koin.

    ROBUSTNESS (fase pengujian #10): setiap siklus dibungkus try/except —
    kegagalan tak terduga (mis. data korup, bug dependensi) dicatat dan
    siklus berikutnya tetap berjalan; loop tidak pernah mati senyap.
    """
    _deadline = time.time() + (max_minutes * 60 if max_minutes > 0 else 1e12)
    broker = PaperBroker()
    brain = load_brain()
    strategies = load_portfolio()
    # benih baru: turunkan keberanian otak ke strategi yang belum punya
    # riwayat evolusi (entry_thr masih default) agar konsisten satu arah.
    for s in strategies:
        if abs(s.entry_thr - 0.20) < 1e-9 and brain["entry_threshold"] < 0.20:
            s.entry_thr = brain["entry_threshold"]
    scanner = CoinScanner(strategies, brain, interval_min=interval_min)
    scalper = Scalper(strategies)   # jalur kedua: scalping candle 15m

    pg = portfolio_gates(strategies)
    print(f"\n{'=' * 62}\n"
          f"  AI TRADER MULTI-COIN — paper trading | scan pasar IDR\n"
          f"{'=' * 62}")
    print(f"Kas awal : Rp {broker.cash_idr:,.0f} | posisi maks: {MAX_POSITIONS}")
    print(f"Strategi : {len(strategies)} | GERBANG ADAPTIF MILIK AI: "
          f"score>={pg['score_min']:.2f} floor>={pg['signal_floor']:.2f} "
          f"scalp>={pg['scalp_strong']:.2f}/{pg['scalp_floor']:.2f} "
          f"(bukan template — hasil evolusi & eksplorasi bot sendiri)")
    print(f"Jalur    : TREN candle {interval_min}m + SCALP candle "
          f"{SCALP_INTERVAL_MIN}m (maks {SCALP_MAX_POSITIONS} slot scalp)")
    print(f"Brain    : thr={brain['entry_threshold']:.2f} "
          f"SL={brain['stop_loss_pct']:.2%} TP={brain['take_profit_pct']:.2%} "
          f"alloc={brain['alloc_frac']:.0%}\n")

    cycle = 0
    diag = CycleDiag()
    had_error = False          # siklus terakhir gagal? (untuk grace period)
    try:
        while time.time() < _deadline:
            cycle += 1
            had_error = False
            try:
                _scan_cycle(broker, brain, strategies, scanner, diag, cycle,
                            interval_min, deep_n, min_score, min_vol_idr,
                            evolve_every, scalper=scalper)
                strategies = scanner.strategies   # mungkin berubah pasca-evolusi
            except KeyboardInterrupt:
                raise
            except Exception as e:
                import traceback
                had_error = True
                print(f"[ERROR] siklus {cycle} gagal: {type(e).__name__}: {e}")
                traceback.print_exc()
                log_experience({"t": time.time(), "type": "cycle_error",
                                "cycle": cycle,
                                "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print("[ERROR] melanjutkan ke siklus berikutnya...")
            # Jeda adaptif + cek deadline SEBELUM tidur — mencegah run baru
            # dibatalkan (cancel-in-progress) saat hampir selesai, yang dulu
            # membuat snapshot state hilang & loop berkejaran dengan runner.
            # Grace period 15s: jika siklus gagal, beri kesempatan minimal satu
            # siklus perbaikan sebelum berhenti (penting untuk CI).
            now = time.time()
            remaining = _deadline - now
            grace = 15 if had_error else 0
            if remaining <= poll_sec + 60 - grace:
                print(f"[LOOP] mendekati batas waktu ({remaining:.0f}s tersisa)"
                      " — berhenti rapi agar state tersimpan.")
                break
            time.sleep(poll_sec)
    except KeyboardInterrupt:
        print("\nDihentikan. Menyimpan state...")
    finally:
        save_brain(brain)
        save_portfolio(scanner.strategies)
        broker.save()


def _scan_cycle(broker, brain, strategies, scanner, diag, cycle,
                interval_min, deep_n, min_score, min_vol_idr, evolve_every,
                scalper=None):
    """Satu siklus penuh: sweep -> filter -> manajemen posisi -> entry -> evolusi.

    AMBANG = MILIK AI. Konstanta manusia (DEFAULT_MIN_SCORE/SIGNAL_FLOOR) hanya
    fallback; nilai aktif diambil dari gerbang adaptif portofolio (hasil
    evolusi backtest) & otak (brain.json), plus safety-net kesunyian yang
    membuat bot berani mencoba lagi setelah terlalu lama tanpa pengalaman."""
    diag.reset()
    # ---- gerbang adaptif: rata-rata bobot ambang tiap strategi + otak ----
    pg = portfolio_gates(strategies)
    eff_score = float(np.clip(min(brain.get("score_min", pg["score_min"]),
                                  pg["score_min"], DEFAULT_MIN_SCORE), 0.35, 0.62))
    if min_score is not None and min_score != DEFAULT_MIN_SCORE:
        eff_score = min_score          # override eksplisit CLI tetap dihormati
    eff_floor = max(brain.get("signal_floor", pg["signal_floor"]),
                    pg["signal_floor"])
    brain.setdefault("_silence_anchor", time.time())
    for l in apply_silence_decay(brain, strategies, broker):
        print(f"  [BELAJAR] {l}")
        log_experience({"t": time.time(), "type": "silence_decay", "lesson": l})
    diag._min_score = eff_score
    ts = time.strftime("%H:%M:%S")
    universe, src = get_universe(min_vol_idr=min_vol_idr)
    diag.market(len(universe), src)
    results = scanner.scan(universe, deep_n=deep_n, diag=diag)
    # [FILTER] gerbang skor+tren+konfirmasi kini terpadu & berlog di scanner
    opps = opportunities(results, eff_score, diag=diag, signal_floor=eff_floor)
    for r in results:
        if r["score"] >= eff_score and r["uptrend"]:
            diag.score_pass_one()

    # ---- JALUR KEDUA: scalping candle 15 menit (peluang jangka pendek) ----
    scalp_results, scalp_opps = [], []
    if scalper is not None:
        scalp_results = scalper.scan(universe, deep_n=max(deep_n, 10), diag=diag)
        scalp_opps = scalper.scalp_opportunities(
            scalp_results, diag=diag,
            strong_thr=pg["scalp_strong"], floor_thr=pg["scalp_floor"])
        scalper.strategies = strategies      # sinkron pasca-evolusi
    scalp_pairs = {r["pair"] for r in scalp_results}

    # harga live Indodax untuk semua pair yang relevan (posisi + kandidat)
    pairs = ({p["pair"] for p in results} | scalp_pairs
             | set(broker.positions))
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
        is_scalp = pos.entry_features.get("_scalp")
        sig_now = None
        res = next((x for x in (results if not is_scalp else scalp_results)
                    if x["pair"] == pair), None)
        if res:
            sig_now = res["signal"]
        else:
            # posisi tanpa kandidat di papan (mis. pair lama) -> re-fetch ringkas
            df, _ = fetch_ohlcv_or_synthetic(
                pair, period=str(SCALP_INTERVAL_MIN if is_scalp else interval_min),
                limit=300)
            if df is not None and len(df) >= 60:
                f2 = normalize_signal_features(build_features(df))
                sig_now = combined_signal(strategies, f2, len(f2) - 1)
        if is_scalp:
            # exit khusus scalp: TP/SL ATR-adaptif, time-stop, trailing ketat
            reason = scalp_exit_decision(pos, price, sig_now)
            if reason:
                trade = broker.sell(pair, price, reason)
                if trade:
                    lessons = learn_from_trade(trade, strategies, brain)
                    save_brain(brain)
                    save_portfolio(strategies)
                    print(f"  >> JUAL-SCALP {pair.split('_')[0].upper()} "
                          f"({reason}) PnL: Rp {trade['pnl']:,.0f} "
                          f"({trade['pnl_pct']:+.2%})")
                    for l in lessons:
                        print(f"     [BELAJAR] {l}")
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
            # FIX BUG "exit tersandera": dulu ambang signal_exit ikut threshold
            # entry yang dinaikkan AI (thr=0.45 -> butuh sig<-0.225, hanya 0.8%
            # dari bar). Posisi tren tak pernah bisa keluar via sinyal -> menggantung
            # sampai SL/trailing penuh (WR trend 7%, EV -Rp26.800/trade).
            # Exit adalah keputusan terpisah dari entry: pakai ambang tetap yang
            # longgar; TP/SL/trailing tetap menjadi pengaman utama.
            if sig_now is not None and sig_now < -SIGNAL_EXIT_FLOOR:
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
    print(f"[ENTRY] kandidat lolos semua gerbang skor+tren+konfirmasi: {len(opps)}")
    for opp in opps:                      # sudah terurut skor desc
        pair = opp["pair"]
        sym = pair.split("_")[0].upper()
        if len(broker.positions) >= MAX_POSITIONS:
            diag.reject(sym, "risk_management", "max_positions_reached",
                        actual=f"{len(broker.positions)}/{MAX_POSITIONS}")
            break
        if pair in held:
            continue
        # otak memilih sendiri ambang entry (adaptif, hasil evolusi).
        # FIX "reject >> open": gerbang scanner (floor) dan gerbang konfirmasi
        # entry (thr) sebelumnya bisa bertentangan — kandidat yang sudah
        # dinyatakan LOLOS oleh scanner gugur lagi di sini hanya karena AI
        # sedang dalam fase belajar dengan ambang lebih tinggi. Scanner adalah
        # SATU-SATUNYA penjaga gerbang peluang; di tahap eksekusi kita hormati
        # ambang belajar AI HANYA bila ia justru lebih longgar dari floor
        # (thr = min), tidak pernah memperketat kembali secara senyap.
        thr = min(brain["entry_threshold"], eff_floor if eff_floor > 0 else SIGNAL_FLOOR)
        if opp["signal"] < thr:
            diag.reject(sym, "strategy_confirmation",
                        "signal_below_entry_floor",
                        actual=f"signal={opp['signal']:+.3f}",
                        required=f">={thr:.2f}")
            continue      # konfirmasi arah: sinyal gabungan tidak boleh negatif kuat
        price = prices.get(pair) or opp["last_close"]
        feats_snap = {k: v for k, v in opp["features"].items()}
        feats_snap["_scan_score"] = opp["score"]
        diag.entry_ready_one()
        trade = broker.buy(pair, price, feats_snap, alloc=brain["alloc_frac"])
        if trade:
            held.add(pair)
            diag.execution(sym, True,
                           f"@{price:,.0f} skor={opp['score']:.2f} "
                           f"sinyal={opp['signal']:+.2f} "
                           f"mom24h={opp['chg24h_pct']:+.1f}% qty={trade['qty']:.6g}")
            print(f"  >> BELI {sym} @ {price:,.0f} "
                  f"(skor={opp['score']:.2f} sinyal={opp['signal']:+.2f} "
                  f"mom24h={opp['chg24h_pct']:+.1f}%)")
        else:
            diag.execution(sym, False, broker.last_reject or "unknown")

    # ---- entry SCALP (timeframe 15m): slot terpisah dari tren ----
    if scalp_opps:
        n_scalp = sum(1 for p in broker.positions.values()
                      if p.entry_features.get("_scalp"))
        print(f"[ENTRY-SCALP] kandidat lolos gerbang scalp 15m: {len(scalp_opps)}"
              f" | slot scalp aktif: {n_scalp}/{SCALP_MAX_POSITIONS}")
        for opp in scalp_opps:
            pair = opp["pair"]
            sym = pair.split("_")[0].upper()
            if pair in held:
                continue
            if len(broker.positions) >= MAX_POSITIONS:
                diag.reject(sym, "risk_management", "max_positions_reached",
                            actual=f"{len(broker.positions)}/{MAX_POSITIONS}")
                break
            if n_scalp >= SCALP_MAX_POSITIONS:
                diag.reject(sym, "risk_management", "scalp_slot_full",
                            actual=f"{n_scalp}/{SCALP_MAX_POSITIONS}")
                break
            price = prices.get(pair) or opp["last_close"]
            feats_snap = dict(opp["features"])
            feats_snap["_scalp"] = True
            feats_snap["_scalp_atr_pct"] = opp["atr_pct"]
            feats_snap["_scan_score"] = opp["signal"]
            diag.entry_ready_one()
            trade = broker.buy(pair, price, feats_snap,
                               alloc=min(brain["alloc_frac"], 0.25))
            if trade:
                held.add(pair)
                n_scalp += 1
                diag.execution(sym, True,
                               f"SCALP@{price:,.0f} sig={opp['signal']:+.2f} "
                               f"mom5={opp['mom5']:+.2%} atr={opp['atr_pct']:.2%}")
                print(f"  >> BELI-SCALP {sym} @ {price:,.0f} "
                      f"(sig={opp['signal']:+.2f} mom5={opp['mom5']:+.2%} "
                      f"atr={opp['atr_pct']:.2%})")
            else:
                diag.execution(sym, False, broker.last_reject or "unknown")

    # ---- tampilkan papan peringkat peluang ----
    for r in results[:5]:
        mark = "*" if r["pair"] in broker.positions else " "
        print(f"     {mark}{r['symbol']:<12} skor={r['score']:.2f} "
              f"sig={r['signal']:+.2f} m20={r['mom20']:+.2%} "
              f"vol24h=Rp{r['vol_idr_24h'] / 1e9:.1f}M "
              f"{'UPTREND' if r['uptrend'] else ''}")

    diag.summary(cycle, len(broker.positions), MAX_POSITIONS, broker.cash_idr)

    # ---- evolusi mandiri berkala (backtest BTC sebagai proksi) ----
    # waktu evolusi terakhir dilacak pada scanner._last_evolve agar bertahan
    # antar-siklus tanpa mengubah perilaku asli (interval evolve_every menit).
    if time.time() - getattr(scanner, "_last_evolve", time.time()) > evolve_every * 60:
        print("  >> AI melakukan evolusi strategi (self-improvement)...")
        # multi-koin: BTC + ETH sebagai proksi pasar. FIX AUDIT FASE 4: dulu
        # kedua frame di-concat TANPA penyelarasan waktu menjadi SATU deret
        # harga — menghasilkan lonjakan palsu di sambungan (bukti repro: +2,7%)
        # yang mencemari semua fitur rolling & backtest. Sekarang tiap aset
        # dievolusikan pada DERET-NYA SENDIRI (walk-forward train/OOS internal),
        # lalu kandidat terbaik dipilih lewat penjumlahan fitness antar-aset —
        # tidak ada satu pun urutan harga campur-adukan.
        import pandas as pd
        frames = []
        for p in ("btc_idr", "eth_idr"):
            d, live = fetch_ohlcv_or_synthetic(p, period=str(interval_min), limit=500)
            if d is not None and len(d) >= 120 and live:
                frames.append((p, d.reset_index(drop=True)))
        if not frames:
            df0, _ = fetch_ohlcv_or_synthetic("btc_idr", period=str(interval_min), limit=500)
            frames = [("btc_idr", df0.reset_index(drop=True))]
        new_strat, res = scanner.strategies, None
        agg = {}
        for name, dframe in frames:
            feats = normalize_signal_features(build_features(dframe))
            cand, r = evolve(scanner.strategies, dframe, feats, brain)
            # kumpulkan metrik per aset untuk laporan jujur
            agg[name] = {k: r.get(k) for k in
                         ("return_pct", "trades", "max_dd_pct", "fitness",
                          "accepted", "oos_return_pct", "oos_trades")}
            if res is None or (r.get("accepted") and r["fitness"] > res.get("fitness", -1e9)):
                new_strat, res = cand, r
        if res is None:
            res = {"fitness": float("nan"), "return_pct": float("nan"),
                   "win_rate": 0.0, "trades": 0}
        res["per_asset"] = agg
        save_portfolio(new_strat)
        save_brain(brain)
        scanner.strategies = new_strat
        strategies[:] = new_strat          # refleksi ke pemanggil siklus
        if scalper is not None:
            scalper.strategies = new_strat   # scalp ikut strategi hasil evolusi
        scanner._last_evolve = time.time()
        print(f"     hasil evolusi: return={res['return_pct']:+.2f}% "
              f"winrate={res['win_rate']:.0%} trades_bt={res.get('trades', 0)} "
              f"thr_belajar={res.get('entry_thr_used', float('nan')):.2f} "
              f"fitness={res['fitness']:.2f} accepted={res.get('accepted')} "
              f"oos_ret={res.get('oos_return_pct', float('nan')):+.2f}% "
              f"per-aset={ {k: round(v['return_pct'], 2) for k, v in agg.items() if v.get('return_pct') is not None} }")


def do_list(deep_n: int, min_score: float, min_vol_idr: float, interval_min: int):
    """Satu kali scan: tampilkan daftar koin dengan peluang tren naik.

    Kini memakai gerbang terpadu yang SAMA dengan loop trading
    (skor >= ambang + uptrend + konfirmasi sinyal strategi >= SIGNAL_FLOOR) plus
    diagnostic logging — sehingga "PELUANG" di sini benar-benar kandidat
    yang bisa dieksekusi, bukan hanya lolos skor aditif.
    """
    brain = load_brain()
    strategies = load_portfolio()
    scanner = CoinScanner(strategies, brain, interval_min=interval_min)
    scalper = Scalper(strategies)   # jalur kedua: scalping candle 15m
    # tampilkan gerbang adaptif AI yang AKTIF (sama dgn loop trading)
    pg = portfolio_gates(strategies)
    eff_score = float(np.clip(min(brain.get("score_min", pg["score_min"]),
                                  pg["score_min"], DEFAULT_MIN_SCORE), 0.35, 0.62))
    if min_score != DEFAULT_MIN_SCORE:
        eff_score = min_score
    eff_floor = max(brain.get("signal_floor", pg["signal_floor"]), pg["signal_floor"])
    universe, src = get_universe(min_vol_idr=min_vol_idr)
    diag = CycleDiag()
    diag.reset()
    diag._min_score = eff_score
    diag.market(len(universe), src)
    print(f"Memindai {len(universe)} pasar IDR (sumber: {src})... "
          f"[gerbang belajar AI: skor>={eff_score:.2f} floor>={eff_floor:.2f}]")
    results = scanner.scan(universe, deep_n=deep_n, diag=diag)
    opps = opportunities(results, eff_score, diag=diag, signal_floor=eff_floor)
    for r in results:
        if r["score"] >= eff_score and r["uptrend"]:
            diag.score_pass_one()
    print(f"\n{'KOIN':<12} {'SKOR':>6} {'SINYAL':>8} {'MOM24H':>8} {'MOM20B':>8} "
          f"{'VOL24H(IDR)':>14} TREN")
    print("-" * 70)
    for r in results:
        flag = "SIAP-ENTRY ✓" if any(o is r for o in opps) else \
               ("NAIK" if r["uptrend"] and r["score"] >= eff_score else
                ("netral" if r["uptrend"] else "turun"))
        print(f"{r['symbol']:<12} {r['score']:>6.2f} {r['signal']:>+8.2f} "
              f"{r['chg24h_pct']:>7.1f}% {r['mom20']:>+7.2%} "
              f"{r['vol_idr_24h']:>14,.0f} {flag}")
    # jalur scalp 15m — peluang jangka pendek
    scalp_results = scalper.scan(universe, deep_n=max(deep_n, 10), diag=diag)
    scalp_opps = scalper.scalp_opportunities(
        scalp_results, diag=diag,
        strong_thr=pg["scalp_strong"], floor_thr=pg["scalp_floor"])
    if scalp_results:
        print(f"\n-- SCALP 15m: {len(scalp_results)} kandidat dianalisis, "
              f"{len(scalp_opps)} lolos gerbang scalp --")
        for r in sorted(scalp_results, key=lambda x: -x["signal"])[:6]:
            ok = "✓SIAP" if r in scalp_opps else " "
            print(f"     {ok:<6}{r['symbol']:<12} sig={r['signal']:+.3f} "
                  f"mom5={r['mom5']:+.2%} atr={r['atr_pct']:.2%} "
                  f"volSup={r['vol_support']:+.2f}")
    if scalp_opps:
        print(f"=> {len(scalp_opps)} PELUANG SCALP SIAP-ENTRY "
              f"(gerbang belajar AI: sig>={pg['scalp_strong']:.2f} ATAU "
              f"(>={pg['scalp_floor']:.2f} & mom12>0 & EMA>0), volume mendukung, chg24h<=18%)")

    print(f"\n=> {len(opps)} PELUANG SIAP-ENTRY (skor >= {eff_score:.2f}, "
          f"uptrend, DAN sinyal >= {eff_floor:.2f}) [ambang hasil belajar AI]: "
          + (", ".join(o["symbol"] for o in opps) or "(kosong)"))
    diag.summary(1, len(opps), MAX_POSITIONS, 0.0)


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
    ap.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                    help=f"ambang skor peluang tren naik (default {DEFAULT_MIN_SCORE})")
    ap.add_argument("--min-vol", type=float, default=500_000_000,
                    help="volume 24 jam minimum (IDR) agar koin dilirik")
    ap.add_argument("--max-minutes", type=float, default=0.0,
                    help="hentikan loop otomatis setelah N menit (0 = tak terbatas)")
    a = ap.parse_args()

    if a.cmd == "run":
        run_loop(a.pair, a.interval, a.poll, a.evolve_every,
                 scan_mode=(a.pair.lower() in ("", "all", "scan")),
                 deep_n=a.deep, min_score=a.min_score, min_vol_idr=a.min_vol,
                 max_minutes=a.max_minutes)
    elif a.cmd == "scan":
        run_scan_loop(a.interval, a.poll, a.evolve_every,
                      a.deep, a.min_score, a.min_vol, a.max_minutes)
    elif a.cmd == "list":
        do_list(a.deep, a.min_score, a.min_vol, a.interval)
    elif a.cmd == "evolve":
        do_evolve(a.pair, a.interval)
    elif a.cmd == "status":
        do_status()


if __name__ == "__main__":
    main()
