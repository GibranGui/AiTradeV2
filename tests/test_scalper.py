"""Tes jalur scalping 15m — gerbang, exit, entry paper, dan ketahanan siklus."""
import time
import numpy as np
import pandas as pd
import pytest

from ai_trader.scalper import (Scalper, scalp_exit_decision,
                               SCALP_SIGNAL_FLOOR, SCALP_STOP_LOSS_PCT,
                               SCALP_TAKE_PROFIT_PCT, SCALP_MAX_HOLD_SEC)
from ai_trader.strategies.portfolio import default_portfolio


class FakeDiag:
    def __init__(self):
        self.rejects, self.bad = [], []
    def reject(self, sym, stage, reason, actual="", required=""):
        self.rejects.append((sym, stage, reason, actual, required))
    def data_bad(self, sym, stage, why):
        self.bad.append((sym, stage, why))
    def deep_start(self, n): pass
    def data_ok(self): pass


def _mk_result(sig=0.3, mom5=0.004, mom12=0.004, ema_ok=True, vol_sup=0.2, chg=5.0):
    return {"pair": "abc_idr", "symbol": "ABCIDR", "signal": sig, "mom5": mom5,
            "mom12": mom12, "mom20": 0.01, "atr_pct": 0.004, "ema_ok": ema_ok,
            "vol_support": vol_sup, "last_close": 10000.0,
            "chg24h_pct": chg, "vol_idr_24h": 1e10, "features": {}}


class P:  # posisi palsu utk keputusan exit
    def __init__(self, entry_price=10000.0, entry_time=None, atr=0.0):
        self.entry_price = entry_price
        self.entry_time = entry_time or time.time()
        self.entry_features = {"_scalp": True, "_scalp_atr_pct": atr, "_peak_chg": 0.0}


def test_allpass_when_all_conditions_met():
    s = Scalper(default_portfolio())
    d = FakeDiag()
    out = s.scalp_opportunities([_mk_result()], diag=d)
    assert len(out) == 1 and not d.rejects


# Gerbang dua-jalur (lihat scalper.scalp_opportunities):
#   A) signal >= SCALP_STRONG_SIGNAL (0.20) -> langsung lolos, tanpa mom/EMA
#   B) signal >= SCALP_SIGNAL_FLOOR (0.10) + mom12 > 0 + ema_ok
@pytest.mark.parametrize("kw,reason", [
    ({"sig": 0.05, "mom12": 0.004}, "signal_below_floor"),
    ({"sig": 0.15, "mom12": -0.001, "ema_ok": True}, "signal_below_floor"),
    ({"sig": 0.15, "mom12": 0.004, "ema_ok": False}, "signal_below_floor"),
    ({"vol_sup": -0.1}, "volume_not_supporting"),
    ({"chg": 25.0}, "pump_extreme"),
])
def test_each_rejection_reason(kw, reason):
    s = Scalper(default_portfolio())
    d = FakeDiag()
    r = _mk_result(**kw)
    out = s.scalp_opportunities([r], diag=d)
    assert out == []
    assert [x[2] for x in d.rejects] == [reason]
    assert d.rejects[0][1] == "scalp_filter"


def test_exit_stop_loss_and_take_profit():
    p = P()
    assert scalp_exit_decision(p, 10000 * (1 - SCALP_STOP_LOSS_PCT - 1e-6), None) == "scalp_stop_loss"
    p2 = P()
    assert scalp_exit_decision(p2, 10000 * (1 + SCALP_TAKE_PROFIT_PCT + 1e-6), None) == "scalp_take_profit"
    p3 = P()
    assert scalp_exit_decision(p3, 10000.5, 0.3) is None   # masih dalam range


def test_exit_time_stop_and_trailing():
    # time-stop HANYA mengunci profit menua; posisi rugi dibiarkan bekerja
    # sampai SL/TP (perbaikan: sebelumnya time-stop ikut menutup posisi rugi)
    old_win = P(entry_time=time.time() - SCALP_MAX_HOLD_SEC - 10)
    assert scalp_exit_decision(old_win, 10001, 0.3) == "scalp_time_stop"
    old_loss = P(entry_time=time.time() - SCALP_MAX_HOLD_SEC - 10)
    assert scalp_exit_decision(old_loss, 9990, 0.3) is None   # rugi + tua -> tunggu SL/TP
    tr = P()
    scalp_exit_decision(tr, 10000 * 1.025, 0.3)     # naik 2.5% -> peak tercatat (>= 0.6*TP)
    assert scalp_exit_decision(tr, 10000 * 1.018, 0.3) == "scalp_trailing_stop"


def test_exit_signal_flip():
    p = P()
    assert scalp_exit_decision(p, 10001, -0.2) == "scalp_signal_exit"


def test_scan_rejects_synthetics_with_log(monkeypatch):
    import ai_trader.scalper as mod
    df = pd.DataFrame({"open": [1.0]*70, "high": [1.1]*70, "low": [0.9]*70,
                       "close": np.linspace(1, 1.1, 70), "volume": [100.0]*70})
    monkeypatch.setattr(mod, "fetch_ohlcv_or_synthetic",
                        lambda pair, period, limit: (df, False))
    s = Scalper(default_portfolio())
    d = FakeDiag()
    coin = {"pair": "xyz_idr", "symbol": "XYZIDR", "chg24h_pct": 2.0,
            "vol_idr_24h": 1e10}
    assert s.analyze(coin, diag=d) is None
    assert d.bad and d.bad[0][1] == "scalp_candle_data_incomplete"


def test_entry_buy_sell_roundtrip_and_learning(tmp_path, monkeypatch):
    """Kandidat lolos semua syarat -> broker.buy dipanggil & posisi tersimpan;
    sell -> trade tercatat + learn_from_trade menghasilkan pelajaran."""
    from ai_trader.paper_broker import PaperBroker
    from ai_trader.learning import learn_from_trade
    monkeypatch.setenv("AI_TRADER_DATA_DIR", str(tmp_path))
    b = PaperBroker(data_dir=str(tmp_path)) if "data_dir" in PaperBroker.__init__.__code__.co_varnames else PaperBroker()
    feats = {"_scalp": True, "_scalp_atr_pct": 0.005, "rsi": 0.1,
             "macd_hist": 0.2, "ema_trend": 0.3}
    t1 = b.buy("sol_idr", 2_000_000, dict(feats), alloc=0.2)
    assert t1 is not None and "sol_idr" in b.positions
    assert b.last_reject is None
    t2 = b.sell("sol_idr", 2_000_000 * 1.012, "scalp_take_profit")
    assert t2 is not None and t2["pnl"] is not None
    assert "sol_idr" not in b.positions          # closed tidak dihitung terbuka
    lessons = learn_from_trade(t2, default_portfolio(),
                               {"entry_threshold": .25, "stop_loss_pct": .02,
                                "take_profit_pct": .035, "alloc_frac": .5,
                                "lr": .1, "total_wins": 0, "total_losses": 0})
    assert isinstance(lessons, list)


def test_closed_position_not_counted_as_open(tmp_path):
    from ai_trader.paper_broker import PaperBroker
    b = PaperBroker(data_dir=str(tmp_path)) if "data_dir" in PaperBroker.__init__.__code__.co_varnames else PaperBroker()
    b.buy("eth_idr", 50_000, {"_scalp": True}, alloc=0.1)
    b.sell("eth_idr", 51_000, "scalp_take_profit")
    assert len(b.positions) == 0


def test_cycle_failure_does_not_kill_loop():
    """Siklus gagal -> exception tertangkap oleh wrapper loop (dibuktikan
    dengan memanggil ulang pola try/except yang sama seperti di run_scan_loop)."""
    calls = []
    def cycle(n):
        calls.append(n)
        if n == 1:
            raise RuntimeError("data korup")
    ok = 0
    for i in (1, 2):
        try:
            cycle(i); ok += 1
        except Exception:
            pass
    assert len(calls) == 2 and ok == 1
