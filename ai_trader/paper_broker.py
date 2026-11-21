"""Paper broker: simulasi perdagangan dengan harga pasar ASLI dari Indodax.

Tidak ada uang sungguhan — hanya saldo virtual yang dieksekusi pada
ticker/last price real-time Indodax, lengkap dengan fee exchange (0,3%).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

from .market_data import get_ticker

WALLET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "paper_wallet.json")
FEE = 0.003  # fee maker Indodax 0,3%

# ============================================================================
# MODE REAL (uang sungguhan) — DINONAKTIFKAN SECARA DEFAULT.
# Untuk mengaktifkan: set env LIVE_TRADING=1 DAN konfirmasi tertulis dengan
# membuat file data/LIVE_CONFIRM.txt berisi persis kata "SAYA PAHAM RISIKONYA".
# API key dibaca dari env: INDODAX_API_KEY / INDODAX_API_SECRET.
# WAJIB: buat API key Indodax HANYA dengan izin "Trade", JANGAN "Withdraw".
# ============================================================================
LIVE_MODE = os.environ.get("LIVE_TRADING") == "1" and os.path.exists(
    os.path.join(os.path.dirname(__file__), "..", "data", "LIVE_CONFIRM.txt")
)


@dataclass
class Position:
    pair: str
    qty: float          # jumlah koin (BTC)
    entry_price: float
    entry_time: float
    entry_features: dict  # snapshot fitur saat masuk — dipakai untuk belajar
    cost_idr: float = 0.0   # TOTAL kas yang benar-benar keluar saat BUY
                            # (termasuk fee) = cost basis. Default 0 ->
                            # dihitung dari qty*entry_price utk state lama.


MAX_POSITIONS = 5   # multi-coin: maksimal koin yang dipegang bersamaan


class PaperBroker:
    """Dompet paper trading MULTI-KOIN.

    Dulu hanya memegang satu posisi; kini bisa memegang beberapa altcoin
    sekaligus (dibatasi MAX_POSITIONS), masing-masing dengan slot kas sendiri.
    Format lama (satu posisi) otomatis dimigrasikan saat wallet dibaca.
    """

    def __init__(self, start_idr: float = 10_000_000, wallet_path: str = WALLET_PATH):
        self.wallet_path = wallet_path
        self.cash_idr = start_idr
        self.positions: dict[str, Position] = {}   # pair -> Position
        self.trades: list[dict] = []
        self.last_reject: str | None = None   # alasan penolakan buy terakhir
        self._load()

    # properti tunggal untuk kompatibilitas kode lama: posisi pertama (atau None)
    @property
    def position(self) -> Position | None:
        return next(iter(self.positions.values()), None)

    # ---------- persistensi ----------
    def _load(self):
        if os.path.exists(self.wallet_path):
            try:
                with open(self.wallet_path) as f:
                    w = json.load(f)
                self.cash_idr = w["cash_idr"]
                self.trades = w.get("trades", [])
                pos_list = w.get("positions")
                if pos_list is None:                       # format lama: satu objek
                    p = w.get("position")
                    pos_list = [p] if p else []
                self.positions = {p["pair"]: Position(**p) for p in pos_list}
            except Exception:
                pass

    def save(self):
        """Tulis ATOMIK (tmp + fsync + rename).

        FIX audit Fase 6: penulisan langsung ke file tujuan dapat menghasilkan
        JSON parsial bila proses mati di tengah tulis (CI timeout / pembatalan
        run) — _load lalu gagal parse dan DOMPET KEMBALI KE AWAL (posisi & kas
        hilang senyap). Rename POSIX bersifat atomik: pembaca tidak pernah
        melihat file setengah tertulis.
        """
        os.makedirs(os.path.dirname(self.wallet_path), exist_ok=True)
        w = {
            "cash_idr": self.cash_idr,
            "positions": [asdict(p) for p in self.positions.values()],
            "position": None,                              # field warisan (deprecated)
            "trades": self.trades[-500:],
        }
        tmp = self.wallet_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(w, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.wallet_path)

    # ---------- harga ----------
    def current_price(self, pair: str) -> float | None:
        t = get_ticker(pair)
        return t["last_price"] if t else None

    # ---------- eksekusi ----------
    def buy(self, pair: str, price: float, features: dict, alloc: float = 1.0) -> dict | None:
        """Beli koin tertentu. alloc: fraksi dari KAS TERSISA yang dipakai (0..1).

        BUG FIX (silent failure): setiap penolakan kini disimpan di
        `self.last_reject` beserta alasannya, supaya pemanggil bisa mencatat
        kegagalan eksekusi alih-alih hanya menerima `None` tanpa penjelasan.
        """
        self.last_reject = None
        if pair in self.positions:
            self.last_reject = "already_holding_pair"
            return None
        if price <= 0:
            self.last_reject = f"invalid_price({price})"
            return None
        if len(self.positions) >= MAX_POSITIONS:
            self.last_reject = "max_positions_reached"
            return None
        spend = self.cash_idr * alloc
        if spend < 50_000:  # min trade ~50rb IDR
            self.last_reject = f"insufficient_available_balance(spend={spend:.0f}<50000)"
            return None
        fee = spend * FEE
        qty = (spend - fee) / price
        self.cash_idr -= spend
        self.positions[pair] = Position(pair, qty, price, time.time(), dict(features),
                                        cost_idr=spend)
        rec = {"action": "BUY", "pair": pair, "price": price, "qty": qty,
               "fee_idr": fee, "time": self.positions[pair].entry_time, "pnl": None}
        self.trades.append(rec)
        self.save()
        return rec

    def sell(self, pair: str, price: float, reason: str = "signal") -> dict | None:
        if pair not in self.positions:
            self.last_reject = f"no_open_position({pair})"
            return None
        if price <= 0:
            # jangan pop posisi kalau harga invalid — posisi tetap utuh
            self.last_reject = f"invalid_price({price})"
            return None
        pos = self.positions.pop(pair)
        gross = pos.qty * price
        fee = gross * FEE
        proceeds = gross - fee
        # Cost basis = TOTAL kas yang keluar saat BUY (termasuk fee beli).
        # State lama tanpa field cost_idr -> fallback qty*entry_price.
        cost_basis = pos.cost_idr or (pos.qty * pos.entry_price)
        pnl_idr = proceeds - cost_basis
        pnl_pct = pnl_idr / cost_basis if cost_basis > 0 else 0.0
        self.cash_idr += proceeds
        rec = {"action": "SELL", "pair": pair, "price": price, "qty": pos.qty,
               "fee_idr": fee, "time": time.time(), "pnl": pnl_idr, "pnl_pct": pnl_pct,
               "cost_basis_idr": cost_basis,
               "reason": reason, "entry_features": pos.entry_features,
               "holding_sec": time.time() - pos.entry_time}
        self.trades.append(rec)
        self.save()
        return rec

    def equity(self, prices: dict[str, float] | float) -> float:
        """Ekuitas total. `prices`: dict pair->harga, atau satu harga (mode lama)."""
        eq = self.cash_idr
        if isinstance(prices, dict):
            for pair, pos in self.positions.items():
                px = prices.get(pair, pos.entry_price)
                eq += pos.qty * px * (1 - FEE)
        else:
            for pos in self.positions.values():
                eq += pos.qty * prices * (1 - FEE)
        return eq
