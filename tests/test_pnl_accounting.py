"""TAHAP 2 audit — unit test akuntansi PnL PaperBroker dengan angka eksplisit.

Fee Indodax 0,3% dua sisi. Konvensi yang diuji:
- cost basis = TOTAL kas yang keluar saat BUY (pokok + fee beli), disimpan
  di Position.cost_idr;
- proceeds jual = qty*harga - fee jual;
- pnl_idr = proceeds - cost_basis;
- pnl_pct = pnl_idr / cost_basis (dasar konsisten dengan pnl nominal);
- ekuitas = kas + nilai pasar posisi terbuka setelah biaya exit.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_trader.paper_broker import FEE, MAX_POSITIONS, PaperBroker


def mk(tmp_path, start=10_000_000):
    return PaperBroker(start_idr=start, wallet_path=str(tmp_path / "wallet.json"))


FEAT = {"rsi": 50.0}


def test_buy_sell_flat_price_loss_equals_total_fee(tmp_path):
    """1) BUY lalu SELL tanpa perubahan harga -> rugi persis total fee dua sisi."""
    b = mk(tmp_path)
    buy = b.buy("btc_idr", 100_000.0, FEAT, alloc=0.10)     # spend 1jt
    assert buy is not None
    sell = b.sell("btc_idr", 100_000.0, "test")
    expected_loss = 1_000_000 * FEE + (1_000_000 * (1 - FEE)) * FEE
    assert abs(sell["pnl"] + expected_loss) < 1e-6
    assert abs(sell["pnl_pct"] - (-expected_loss / 1_000_000)) < 1e-9
    # kas kembali = 10jt - total fee
    assert abs(b.cash_idr - (10_000_000 - expected_loss)) < 1e-6


def test_buy_sell_profit(tmp_path):
    """2) BUY 100rb -> SELL 110rb: profit bersih setelah kedua fee."""
    b = mk(tmp_path)
    b.buy("eth_idr", 100_000.0, FEAT, alloc=0.10)           # spend 1jt
    qty = b.positions["eth_idr"].qty                        # (1jt*0.997)/100rb
    sell = b.sell("eth_idr", 110_000.0, "tp")
    proceeds = qty * 110_000.0 * (1 - FEE)
    cost = 1_000_000.0
    assert abs(sell["pnl"] - (proceeds - cost)) < 1e-6
    assert abs(sell["pnl_pct"] - (proceeds - cost) / cost) < 1e-9
    assert sell["pnl"] > 0


def test_buy_sell_loss(tmp_path):
    """3) BUY 100rb -> SELL 90rb: rugi terefleksikan benar."""
    b = mk(tmp_path)
    b.buy("xrp_idr", 100_000.0, FEAT, alloc=0.10)
    sell = b.sell("xrp_idr", 90_000.0, "sl")
    assert sell["pnl"] < 0
    qty = sell["qty"]
    assert abs(sell["pnl"] - (qty * 90_000.0 * (1 - FEE) - 1_000_000.0)) < 1e-6


def test_fees_both_sides_recorded(tmp_path):
    """4) Fee tercatat di jurnal BUY dan SELL, tidak dobel/tidak hilang."""
    b = mk(tmp_path)
    buy = b.buy("ada_idr", 50_000.0, FEAT, alloc=0.20)      # spend 2jt
    assert abs(buy["fee_idr"] - 2_000_000 * FEE) < 1e-9
    sell = b.sell("ada_idr", 55_000.0, "tp")
    assert abs(sell["fee_idr"] - sell["qty"] * 55_000.0 * FEE) < 1e-9
    # total fee == selisih kas vs equity awal pada harga flat check via pnl
    assert abs(sell["pnl"] - (sell["qty"] * 55_000.0 * (1 - FEE) - 2_000_000)) < 1e-6


def test_full_position_only_no_partial(tmp_path):
    """5) Broker full-position: sell sebagian TIDAK didukung (desain)."""
    b = mk(tmp_path)
    b.buy("dot_idr", 100_000.0, FEAT, alloc=0.10)
    pos_qty = b.positions["dot_idr"].qty
    b.sell("dot_idr", 100_000.0, "test")
    assert "dot_idr" not in b.positions          # seluruhnya keluar
    assert abs(b.trades[-1]["qty"] - pos_qty) < 1e-12


def test_open_position_valued_in_equity(tmp_path):
    """6) Posisi belum dijual -> ekuitas memperhitungkan mark-to-market net fee."""
    b = mk(tmp_path)
    b.buy("sol_idr", 100_000.0, FEAT, alloc=0.10)           # spend 1jt
    eq_at_entry = b.equity({"sol_idr": 100_000.0})
    base = 10_000_000 - 1_000_000                            # sisa kas
    gross = b.positions["sol_idr"].qty * 100_000.0
    assert abs(eq_at_entry - (base + gross * (1 - FEE))) < 1e-6
    eq_up = b.equity({"sol_idr": 120_000.0})
    assert eq_up > eq_at_entry


def test_sequential_trades_cash_conservation(tmp_path):
    """7) Banyak transaksi berurutan: kas selalu == 10jt + Σpnl realisasi."""
    b = mk(tmp_path)
    realized = 0.0
    for i, px_out in enumerate([110_000, 95_000, 101_000, 88_000]):
        pair = f"c{i}_idr"
        b.buy(pair, 100_000.0, FEAT, alloc=0.05)            # spend 500rb
        s = b.sell(pair, float(px_out), "seq")
        realized += s["pnl"]
        assert abs(b.cash_idr - (10_000_000 + realized)) < 1e-6


def test_state_recovery_after_restart(tmp_path):
    """8) Restart: posisi (termasuk cost_idr & trailing di features) pulih utuh."""
    b = mk(tmp_path)
    b.buy("link_idr", 100_000.0, {"trail_peak": 101_000.0}, alloc=0.10)
    b2 = PaperBroker(wallet_path=b.wallet_path)
    assert "link_idr" in b2.positions
    p = b2.positions["link_idr"]
    assert abs(p.cost_idr - 1_000_000.0) < 1e-9
    assert p.entry_features["trail_peak"] == 101_000.0
    assert abs(b2.cash_idr - b.cash_idr) < 1e-9
    # sell dari broker hasil-reload memakai cost basis yang sama
    s = b2.sell("link_idr", 105_000.0, "tp")
    assert abs(s["pnl"] - (p.qty * 105_000.0 * (1 - FEE) - 1_000_000.0)) < 1e-6


def test_legacy_state_without_cost_idr(tmp_path):
    """Kompatibilitas: state lama tanpa field cost_idr tetap bisa dimuat."""
    import json
    p = tmp_path / "wallet.json"
    legacy = {
        "cash_idr": 9_000_000.0,
        "positions": [{"pair": "old_idr", "qty": 10.0, "entry_price": 100_000.0,
                       "entry_time": 1.0, "entry_features": {}}],
        "position": None, "trades": [],
    }
    p.write_text(json.dumps(legacy))
    b = PaperBroker(wallet_path=str(p))
    assert "old_idr" in b.positions
    assert b.positions["old_idr"].cost_idr == 0.0
    s = b.sell("old_idr", 100_000.0, "test")
    # fallback cost basis = qty*entry_price -> rugi = fee jual saja
    assert abs(s["pnl"] - (10 * 100_000 * -FEE)) < 1e-6


def test_reject_reasons_and_closed_not_resellable(tmp_path):
    """Silent-failure guard: setiap penolakan punya alasan; pair tertutup tak bisa dijual lagi."""
    b = mk(tmp_path)
    assert b.sell("ghost_idr", 1000.0, "x") is None
    assert "no_open_position" in b.last_reject
    b.buy("btc_idr", 100_000.0, FEAT, alloc=0.10)
    assert b.buy("btc_idr", 100_000.0, FEAT) is None
    assert b.last_reject == "already_holding_pair"
    assert b.sell("btc_idr", -5.0, "x") is None       # harga invalid
    assert "btc_idr" in b.positions                    # posisi TIDAK terpop
    assert "invalid_price" in b.last_reject
    # max positions
    for i in range(MAX_POSITIONS):
        b.buy(f"k{i}_idr", 1000.0, FEAT, alloc=0.01)
    extra = [f"x{i}_idr" for i in range(10)]
    got = sum(1 for x in extra if b.buy(x, 1000.0, FEAT, alloc=0.01))
    assert got == 0 and b.last_reject == "max_positions_reached"
