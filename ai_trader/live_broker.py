"""IndodaxLiveBroker — order sungguhan ke exchange (OPT-IN, default mati).

Lapisan keamanan berlapis:
  1. LIVE_TRADING=1 harus di-set via environment.
  2. File data/LIVE_CONFIRM.txt harus berisi "SAYA PAHAM RISIKONYA".
  3. API key hanya dari env (INDODAX_API_KEY / INDODAX_API_SECRET) — tidak pernah di-commit.
  4. Batas kerugian harian (max_daily_loss_idr) & batas ukuran posisi per trade.
  5. Whitelist pair opsional — bot tidak akan menyentuh coin di luar daftar.
  6. Semua order ditulis ke data/live_audit.log untuk jejak audit.

Cara pakai (setelah minimal 1-3 bulan paper trading profit konsisten):
  export INDODAX_API_KEY=...   # izin TRADE saja, JANGAN WITHDRAW
  export INDODAX_API_SECRET=...
  export LIVE_TRADING=1
  echo "SAYA PAHAM RISIKONYA" > data/LIVE_CONFIRM.txt
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import urllib.parse
from urllib.request import Request, urlopen

API_URL = "https://indodax.com/api"
AUDIT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "live_audit.log")

CONFIRM_PHRASE = "SAYA PAHAM RISIKONYA"


class LiveSafetyError(RuntimeError):
    pass


def _audit(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    try:
        with open(AUDIT_PATH, "a") as f:
            f.write(line)
    except OSError:
        pass
    print("AUDIT:", line.strip())


class IndodaxLiveBroker:
    """Eksekusi order real di Indodax dengan guard-rails ketat."""

    def __init__(
        self,
        max_position_idr: float = 500_000,     # maks Rp per satu order beli
        max_daily_loss_idr: float = 1_000_000, # stop total jika rugi hari ini lewat ini
        allowed_pairs: set[str] | None = None, # None = semua; isi set utk whitelist
    ):
        # --- gerbang keamanan 1 & 2 ---
        if os.environ.get("LIVE_TRADING") != "1":
            raise LiveSafetyError("LIVE_TRADING != 1. Mode real dinonaktifkan.")
        conf = os.path.join(os.path.dirname(__file__), "..", "data", "LIVE_CONFIRM.txt")
        if not os.path.exists(conf) or open(conf).read().strip() != CONFIRM_PHRASE:
            raise LiveSafetyError(
                f'Konfirmasi tertulis belum ada. Buat file {conf} berisi "{CONFIRM_PHRASE}".'
            )
        # --- gerbang 3: kredensial hanya dari env ---
        self.key = os.environ.get("INDODAX_API_KEY", "")
        self.secret = os.environ.get("INDODAX_API_SECRET", "")
        if not self.key or not self.secret:
            raise LiveSafetyError("INDODAX_API_KEY / INDODAX_API_SECRET belum di-set.")

        self.max_position_idr = max_position_idr
        self.max_daily_loss_idr = max_daily_loss_idr
        self.allowed_pairs = allowed_pairs
        self._day = time.strftime("%Y-%m-%d")
        self._loss_today = 0.0
        _audit(f"LIVE broker aktif. Maks/posisi=Rp{max_position_idr:,.0f} "
               f"maks-rugi-harian=Rp{max_daily_loss_idr:,.0f} pairs={allowed_pairs or 'ALL'}")

    # ---------- REST signed ----------
    def _sign(self, params: dict) -> dict:
        query = urllib.parse.urlencode(params)
        sig = hmac.new(self.secret.encode(), query.encode(), hashlib.sha512).hexdigest()
        return {"query": query, "signature": sig}

    def _post(self, endpoint: str, params: dict):
        ts = int(time.time())
        base = {"method": endpoint, "access_key": self.key, "nonce": ts * 1000}
        base.update(params)
        body = self._sign(base)
        req = Request(API_URL + "/private",
                      data=urllib.parse.urlencode(body).encode(),
                      headers={"User-Agent": "AiTradeV2-live/1.0"})
        with urlopen(req, timeout=20) as r:
            out = json.loads(r.read())
        if isinstance(out, dict) and out.get("error"):
            raise LiveSafetyError(f"Order ditolak exchange: {out['error']}")
        return out

    # ---------- guard rails ----------
    def _check_day(self):
        d = time.strftime("%Y-%m-%d")
        if d != self._day:
            self._day, self._loss_today = d, 0.0

    def _guard_pair(self, pair: str):
        if self.allowed_pairs is not None and pair not in self.allowed_pairs:
            raise LiveSafetyError(f"Pair {pair} tidak ada di whitelist live.")

    def _guard_size(self, idr_amount: float):
        if idr_amount > self.max_position_idr:
            _audit(f"Batas ukuran: Rp{idr_amount:,.0f} dipangkas ke Rp{self.max_position_idr:,.0f}")
            return self.max_position_idr
        return idr_amount

    def _guard_daily_loss(self):
        self._check_day()
        if self._loss_today >= self.max_daily_loss_idr:
            raise LiveSafetyError(
                f"Kerugian hari ini Rp{self._loss_today:,.0f} >= batas "
                f"Rp{self.max_daily_loss_idr:,.0f}. Trading dihentikan sampai besok.")

    # ---------- balance ----------
    def get_balances(self) -> dict:
        info = self._post("/trading-info-v1", {})
        bal = {}
        for k, v in info.items():
            if k.endswith("_available"):
                bal[k.replace("_available", "").lower()] = float(v)
        return bal

    # ---------- eksekusi ----------
    def buy(self, pair: str, price: float, idr_amount: float) -> dict | None:
        """Market-buy: belanjakan idr_amount (dibatasi max_position_idr)."""
        self._guard_pair(pair)
        self._guard_daily_loss()
        idr_amount = self._guard_size(idr_amount)
        if idr_amount < 50_000 or price <= 0:
            return None
        qty = round(idr_amount / price, 8)
        _audit(f"BUY market {pair} qty={qty} (~Rp{qty*price:,.0f})")
        res = self._post("/order", {"pair": pair.lower(), "side": "buy", "price": str(int(price)),
                                    "volume": str(qty)})
        return {"action": "BUY", "pair": pair, "qty": qty, "order_id": res}

    def sell(self, pair: str, price: float, qty: float, pnl_idr: float = 0.0) -> dict | None:
        self._guard_pair(pair)
        if qty <= 0:
            return None
        self._check_day()
        if pnl_idr < 0:
            self._loss_today += -pnl_idr
        _audit(f"SELL market {pair} qty={qty} pnl≈Rp{pnl_idr:,.0f} "
               f"(kumulatif rugi hari ini Rp{self._loss_today:,.0f})")
        res = self._post("/order", {"pair": pair.lower(), "side": "sell",
                                    "price": str(int(price)), "volume": str(qty)})
        return {"action": "SELL", "pair": pair, "qty": qty, "order_id": res}
