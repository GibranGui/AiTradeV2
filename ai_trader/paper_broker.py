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


@dataclass
class Position:
    pair: str
    qty: float          # jumlah koin (BTC)
    entry_price: float
    entry_time: float
    entry_features: dict  # snapshot fitur saat masuk — dipakai untuk belajar


class PaperBroker:
    def __init__(self, start_idr: float = 10_000_000, wallet_path: str = WALLET_PATH):
        self.wallet_path = wallet_path
        self.cash_idr = start_idr
        self.position: Position | None = None
        self.trades: list[dict] = []
        self._load()

    # ---------- persistensi ----------
    def _load(self):
        if os.path.exists(self.wallet_path):
            try:
                with open(self.wallet_path) as f:
                    w = json.load(f)
                self.cash_idr = w["cash_idr"]
                p = w.get("position")
                self.position = Position(**p) if p else None
                self.trades = w.get("trades", [])
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.wallet_path), exist_ok=True)
        w = {
            "cash_idr": self.cash_idr,
            "position": asdict(self.position) if self.position else None,
            "trades": self.trades[-500:],
        }
        with open(self.wallet_path, "w") as f:
            json.dump(w, f, indent=2)

    # ---------- harga ----------
    def current_price(self, pair: str) -> float | None:
        t = get_ticker(pair)
        return t["last_price"] if t else None

    # ---------- eksekusi ----------
    def buy(self, pair: str, price: float, features: dict, alloc: float = 1.0) -> dict | None:
        """alloc: fraksi dari kas yang dipakai (0..1)."""
        if self.position is not None or price <= 0:
            return None
        spend = self.cash_idr * alloc
        if spend < 50_000:  # min trade ~50rb IDR
            return None
        fee = spend * FEE
        qty = (spend - fee) / price
        self.cash_idr -= spend
        self.position = Position(pair, qty, price, time.time(), dict(features))
        rec = {"action": "BUY", "pair": pair, "price": price, "qty": qty,
               "fee_idr": fee, "time": self.position.entry_time, "pnl": None}
        self.trades.append(rec)
        self.save()
        return rec

    def sell(self, pair: str, price: float, reason: str = "signal") -> dict | None:
        if self.position is None or self.position.pair != pair or price <= 0:
            return None
        pos = self.position
        gross = pos.qty * price
        fee = gross * FEE
        proceeds = gross - fee
        pnl_idr = proceeds - pos.qty * pos.entry_price
        pnl_pct = price / pos.entry_price - 1 - 2 * FEE
        self.cash_idr += proceeds
        self.position = None
        rec = {"action": "SELL", "pair": pair, "price": price, "qty": pos.qty,
               "fee_idr": fee, "time": time.time(), "pnl": pnl_idr, "pnl_pct": pnl_pct,
               "reason": reason, "entry_features": pos.entry_features,
               "holding_sec": time.time() - pos.entry_time}
        self.trades.append(rec)
        self.save()
        return rec

    def equity(self, price: float) -> float:
        if self.position:
            return self.cash_idr + self.position.qty * price * (1 - FEE)
        return self.cash_idr
