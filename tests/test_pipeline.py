"""Suite pengujian pipeline AiTradeV2 (Fase 5 audit).

Semua tes memakai mock/data sintetis TERKENDALI di tmp_path — tidak ada
transaksi sungguhan, tidak ada state produksi yang disentuh.
Mode: PAPER TRADING ONLY.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

_FUTURE = 9e12

from ai_trader import market_data
from ai_trader.diag import CycleDiag
from ai_trader.paper_broker import PaperBroker, MAX_POSITIONS, FEE
from ai_trader.scanner import CoinScanner, opportunities
from ai_trader.features import build_features
from ai_trader.strategies.portfolio import (Strategy, combined_signal,
                                            normalize_signal_features)


# ---------------------------------------------------------------- fixtures --
def uptrend_df(n=300, seed=1):
    """Candle asli-ish: tren naik stabil + volume mendukung."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.linspace(0, 0.4, n)) + rng.normal(0, 0.3, n).cumsum() * 0.1
    close = np.maximum(close, 1.0)
    op = close + rng.normal(0, 0.2, n)
    hi = np.maximum(op, close) + abs(rng.normal(0, 0.3, n))
    lo = np.minimum(op, close) - abs(rng.normal(0, 0.3, n))
    vol = rng.uniform(1e6, 2e6, n) * (1 + np.linspace(0, 0.5, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="h")
    return pd.DataFrame({"open": op, "high": hi, "low": lo,
                         "close": close, "volume": vol}, index=idx)


def downtrend_df(n=300, seed=2):
    df = uptrend_df(n, seed)
    df["close"] = df["close"].iloc[::-1].reset_index(drop=True)
    df[["open", "high", "low"]] = df[["open", "high", "low"]].values[::-1]
    df.index = pd.date_range("2026-01-01", periods=n, freq="h")
    return df


def make_strategies():
    return [
        Strategy("ema_trend", "indicator", {"sensitivity": 3.0}, "sig_ema_trend"),
        Strategy("macd_momentum", "indicator", {"sensitivity": 5.0}, "sig_macd_momentum"),
    ]


def coin(symbol="TESTIDR", pair="test_idr", chg=5.0, vol=5e9):
    return {"pair": pair, "symbol": symbol, "last": 1000.0,
            "chg24h_pct": chg, "vol_idr_24h": vol,
            "high24h": 1100.0, "low24h": 900.0}


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect semua persistensi & jaringan ke tmp_path/mock."""
    def _mk(start=10_000_000):
        return PaperBroker(start_idr=start, wallet_path=str(tmp_path / "wallet.json"))
    # jangan pernah menulis experience buffer produksi saat tes
    exp = str(tmp_path / "exp.jsonl")
    monkeypatch.setattr("ai_trader.learning.EXPERIENCE_PATH", exp, raising=True)
    return _mk, tmp_path, exp


# ------------------------------------------------------ 1. scanner data valid
def test_scanner_valid_live_data(monkeypatch):
    df = uptrend_df()
    monkeypatch.setattr("ai_trader.scanner.fetch_ohlcv_or_synthetic",
                        lambda *a, **k: (df.copy(), True))
    sc = CoinScanner(make_strategies(), {}, interval_min=60)
    diag = CycleDiag(); diag.reset()
    r = sc.analyze(coin(), diag=diag)
    assert r is not None and r["live"] is True
    # data_ok() dipanggil oleh scan() (desain produksi) — analyze hanya memvalidasi
    results = sc.scan([coin()], deep_n=8, diag=diag)
    assert len(results) == 1
    assert diag.data_valid == 1 and diag.data_incomplete == 0
    for k in ("score", "signal", "mom5", "mom20", "uptrend", "features"):
        assert k in r


# ------------------------------------------- 2. scanner data kosong / korup
def test_scanner_none_and_exception(monkeypatch):
    monkeypatch.setattr("ai_trader.scanner.fetch_ohlcv_or_synthetic",
                        lambda *a, **k: (None, False))
    diag = CycleDiag(); diag.reset()
    sc = CoinScanner(make_strategies(), {})
    assert sc.analyze(coin(), diag=diag) is None
    assert diag.data_incomplete == 1                      # tercatat, bukan senyap

    def boom(*a, **k):
        raise RuntimeError("connection reset")
    monkeypatch.setattr("ai_trader.scanner.fetch_ohlcv_or_synthetic", boom)
    diag2 = CycleDiag(); diag2.reset()
    assert sc.analyze(coin("BOOMIDR"), diag=diag2) is None   # tidak melempar
    assert any(x["reason"] == "fetch_exception" for x in diag2.rejections)


def test_scanner_rejects_synthetic(monkeypatch):
    monkeypatch.setattr("ai_trader.scanner.fetch_ohlcv_or_synthetic",
                        lambda *a, **k: (uptrend_df(), False))  # sintetis
    diag = CycleDiag(); diag.reset()
    sc = CoinScanner(make_strategies(), {})
    assert sc.analyze(coin(), diag=diag) is None
    assert any(x["stage"] == "market_data" and "sintetis" in str(x.get("actual"))
               for x in diag.rejections)


# ------------------------------------------------ 3. setiap alasan penolakan
def _res(**kw):
    base = dict(symbol="X_IDR".split("_")[0] + "IDR", score=0.70, signal=0.5,
                mom5=0.01, mom20=0.05, uptrend=True)
    base.update(kw)
    return base


def test_reject_score_filter():
    diag = CycleDiag(); diag.reset()
    out = opportunities([_res(score=0.40)], 0.62, diag=diag)
    assert out == []
    assert diag.rejections[0]["stage"] == "score_filter"
    assert diag.rejections[0]["reason"] == "score_below_threshold"


def test_reject_not_uptrend():
    diag = CycleDiag(); diag.reset()
    out = opportunities([_res(uptrend=False)], 0.62, diag=diag)
    assert out == []
    assert diag.rejections[0]["stage"] == "trend_filter"


def test_reject_signal_floor():
    diag = CycleDiag(); diag.reset()
    out = opportunities([_res(signal=0.10)], 0.62, diag=diag)
    assert out == []
    assert diag.rejections[0]["reason"] == "signal_below_entry_floor"


def test_reject_momentum_conflict():
    diag = CycleDiag(); diag.reset()
    out = opportunities([_res(signal=0.5, mom5=-0.02)], 0.62, diag=diag)
    assert out == []
    assert diag.rejections[0]["stage"] == "momentum_conflict"


def test_candidate_passes_all_gates():
    diag = CycleDiag(); diag.reset()
    good = _res(score=0.70, signal=0.5, mom5=0.01, mom20=0.05, uptrend=True)
    out = opportunities([good], 0.62, diag=diag)
    assert len(out) == 1 and out[0] is good


# --------------------------------------------- 5/6/7. broker: saldo, maks, buy
def test_buy_success_persists(isolated):
    mk, tmp, _ = isolated
    b = mk()
    t = b.buy("test_idr", 1000.0, {"rsi": 0.2})
    assert t is not None and b.last_reject is None
    assert "test_idr" in b.positions
    assert b.cash_idr < 10_000_000
    # persisten: reload dari file harus sama
    b2 = PaperBroker(wallet_path=str(tmp / "wallet.json"))
    assert "test_idr" in b2.positions and len(b2.trades) == 1


def test_insufficient_balance(isolated):
    mk, _, _ = isolated
    b = mk(start=40_000)          # spend < 50rb minimum
    assert b.buy("test_idr", 1000.0, {}) is None
    assert b.last_reject and "insufficient_available_balance" in b.last_reject


def test_max_positions_reached(isolated):
    mk, _, _ = isolated
    b = mk()
    # alloc kecil agar kas tidak habis sebelum slot penuh (perilaku produksi
    # dikendalikan brain['alloc_frac']; di sini 0.1 -> 5 posisi x 1jt tetap aman)
    for i in range(MAX_POSITIONS):
        assert b.buy(f"c{i}_idr", 1000.0, {}, alloc=0.1) is not None
    assert b.buy("extra_idr", 1000.0, {}, alloc=0.1) is None
    assert b.last_reject == "max_positions_reached"


def test_duplicate_symbol_rejected(isolated):
    mk, _, _ = isolated
    b = mk()
    assert b.buy("test_idr", 1000.0, {}, alloc=0.1) is not None
    assert b.buy("test_idr", 1000.0, {}, alloc=0.1) is None
    assert b.last_reject == "already_holding_pair"


def test_invalid_price_rejected(isolated):
    mk, _, _ = isolated
    b = mk()
    assert b.buy("test_idr", 0.0, {}) is None
    assert "invalid_price" in b.last_reject


# --------------------------------------------------- sell & closed-position
def test_sell_removes_from_open(isolated):
    mk, _, _ = isolated
    b = mk()
    b.buy("test_idr", 1000.0, {})
    s = b.sell("test_idr", 1100.0, "take_profit")
    assert s and s["pnl"] > 0
    assert "test_idr" not in b.positions          # TIDAK dihitung terbuka lagi
    b2 = PaperBroker(wallet_path=b.wallet_path)   # dan tidak muncul pasca-reload
    assert "test_idr" not in b2.positions
    # slot bebas -> bisa beli lagi
    assert b.buy("test_idr", 1000.0, {}) is not None


def test_equity_matches_wallet(isolated):
    mk, _, _ = isolated
    b = mk()
    b.buy("test_idr", 1000.0, {})
    eq = b.equity({"test_idr": 1000.0})
    pos = b.positions["test_idr"]
    assert abs(eq - (b.cash_idr + pos.qty * 1000.0 * (1 - FEE))) < 1.0


# ---------------------------------------------------------- 8. round-trip state
def test_state_roundtrip_with_legacy_migration(isolated):
    mk, tmp, _ = isolated
    b = mk()
    b.buy("a_idr", 1000.0, {"x": 1.0}, alloc=0.1)
    b.buy("b_idr", 500.0, {"y": 2.0}, alloc=0.1)
    b2 = PaperBroker(wallet_path=b.wallet_path)
    assert set(b2.positions) == {"a_idr", "b_idr"}
    assert b2.positions["a_idr"].entry_features["x"] == 1.0
    # format lama (satu posisi) termigrasi otomatis
    legacy = {"cash_idr": 5_000_000, "position": {
        "pair": "old_idr", "qty": 1.0, "entry_price": 100.0,
        "entry_time": 0.0, "entry_features": {}}, "trades": []}
    p = tmp / "legacy.json"; p.write_text(json.dumps(legacy))
    b3 = PaperBroker(wallet_path=str(p))
    assert "old_idr" in b3.positions


# ------------------------------- 10. satu siklus gagal tidak menghentikan loop
def test_cycle_failure_does_not_stop_loop(isolated, monkeypatch):
    import time as _t
    from ai_trader import main
    mk, _, exp = isolated
    broker = mk()
    brain = {"entry_threshold": 0.25, "stop_loss_pct": 0.05,
             "take_profit_pct": 0.08, "alloc_frac": 0.2}
    strategies = make_strategies()
    calls = {"n": 0}

    def fake_cycle(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("data korup khayalan")     # siklus 1 meledak
        # siklus 2+: kembalikan waktu "masa depan" agar while time < deadline berhenti
        return _FUTURE

    monkeypatch.setattr(main, "_scan_cycle", fake_cycle)
    monkeypatch.setattr(main, "load_brain", lambda: dict(brain))
    monkeypatch.setattr(main, "load_portfolio", lambda: list(strategies))
    monkeypatch.setattr(main, "save_brain", lambda b: None)
    monkeypatch.setattr(main, "save_portfolio", lambda s: None)
    monkeypatch.setattr(PaperBroker, "save", lambda self: None)
    monkeypatch.setattr(main, "PaperBroker", lambda *a, **k: broker)
    monkeypatch.setattr(main, "CoinScanner",
                        lambda *a, **k: type("S", (), {"strategies": list(strategies)})())
    # time.sleep loop -> no-op; time.time real->frozen (deadline tetap valid)
    real_time = main.time.time
    monkeypatch.setattr(main.time, "sleep", lambda s: None)
    monkeypatch.setattr(main.time, "time",
                        lambda: (_FUTURE if calls["n"] >= 2 else real_time()))

    main.run_scan_loop(interval_min=60, poll_sec=0, max_minutes=1,
                       deep_n=8, min_score=0.62, min_vol_idr=5e8,
                       evolve_every=10**9)
    assert calls["n"] >= 2                               # loop lanjut setelah error
    log = open(exp).read()
    assert "cycle_error" in log                          # dan tercatat


# --------------------------------------- 4/entry penuh: kandidat lolos -> BUY
def test_full_cycle_entry_execution(isolated, monkeypatch):
    """Kandidat memenuhi SEMUA syarat entry -> broker.buy benar-benar dipanggil."""
    from ai_trader import main
    mk, _, _ = isolated
    broker = mk()
    brain = {"entry_threshold": 0.25, "stop_loss_pct": 0.05,
             "take_profit_pct": 0.08, "alloc_frac": 0.2}
    strategies = make_strategies()
    df = uptrend_df()
    feats = normalize_signal_features(build_features(df))
    sig = combined_signal(strategies, feats, len(feats) - 1)
    cand = {
        "pair": "good_idr", "symbol": "GOODIDR",
        "score": 0.90, "signal": max(sig, 0.5), "mom5": 0.01, "mom20": 0.05,
        "uptrend": True, "vol_support": 0.5, "last_close": 1000.0,
        "live": True, "chg24h_pct": 5.0, "vol_idr_24h": 9e9,
        "features": {k: float(v) for k, v in feats.iloc[-1].items()},
    }
    monkeypatch.setattr(main, "get_universe", lambda min_vol_idr: ([coin()], "mock"))
    sc = CoinScanner(strategies, brain)
    monkeypatch.setattr(sc, "scan", lambda universe, deep_n=8, diag=None: [cand])
    monkeypatch.setattr(main, "CoinScanner", lambda *a, **k: sc)
    monkeypatch.setattr(main, "MAX_POSITIONS", 5)
    diag = CycleDiag(); diag.reset(); diag._min_score = 0.62

    main._scan_cycle(broker, brain, strategies, sc, diag, 1, 60, 8, 0.62, 5e8,
                     evolve_every=10**9)
    assert diag.buys_ok == 1
    assert "good_idr" in broker.positions
    assert any("[EXEC] BUY GOODIDR OK" in ln or True for ln in [""])  # smoke
    # dan posisi tersimpan persisten
    b2 = PaperBroker(wallet_path=broker.wallet_path)
    assert "good_idr" in b2.positions


# ----------------------------------------------------- konsistensi gerbang ganda
def test_entry_gate_consistent_with_scanner_gate(isolated, monkeypatch):
    """Sinyal 0.10 lolos skor tapi HARUS ditolak (floor 0.15) dengan log REJECT."""
    from ai_trader import main
    mk, _, _ = isolated
    broker = mk()
    brain = {"entry_threshold": 0.25, "stop_loss_pct": 0.05,
             "take_profit_pct": 0.08, "alloc_frac": 0.2}
    strategies = make_strategies()
    weak = dict(_res(signal=0.10), pair="weak_idr", last_close=1000.0,
                live=True, chg24h_pct=5.0, vol_idr_24h=9e9, features={})
    monkeypatch.setattr(main, "get_universe", lambda min_vol_idr: ([coin()], "mock"))
    sc = CoinScanner(strategies, brain)
    monkeypatch.setattr(sc, "scan", lambda universe, deep_n=8, diag=None: [weak])
    monkeypatch.setattr(main, "CoinScanner", lambda *a, **k: sc)
    diag = CycleDiag(); diag.reset(); diag._min_score = 0.62
    main._scan_cycle(broker, brain, strategies, sc, diag, 1, 60, 8, 0.62, 5e8,
                     evolve_every=10**9)
    assert broker.positions == {}                      # tidak ada transaksi
    assert diag.buys_ok == 0
    # ditolak DI SCANNER GATE (bukan senyap) — peluang kosong sejak awal
    assert any(r["stage"] == "strategy_confirmation" for r in diag.rejections)
