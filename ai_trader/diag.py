"""Diagnostic logging terstruktur — dikumpulkan untuk diagnosis, bukan untuk
mengubah keputusan strategi.

Format (sesuai standar audit):
  [MARKET]   jumlah pasangan dipindai / data valid / data tidak lengkap
  [FILTER]   lolos volume / indikator / skor / konfirmasi strategi
  [ENTRY]    sinyal BUY, kandidat lolos SEMUA syarat entry
  [REJECT]   per kandidat: simbol, tahap, alasan, nilai aktual vs threshold
  [EXECUTION] pemanggilan broker.buy + hasil/alasan kegagalan + ID posisi
  [SUMMARY]  ringkasan akhir siklus + alasan penolakan paling sering

Logging bersifat aditif: tidak ada satu pun fungsi di sini yang mengubah
nilai kembali ke pemanggil atau memengaruhi logika trading.
"""

from __future__ import annotations

import time


def _p(*args):
    print(*args, flush=True)


class CycleDiag:
    """Penghitung & pelacak penolakan untuk SATU siklus scan."""

    def __init__(self):
        self.reset()

    def reset(self):
        now = time.time()
        self.started = now
        # counters
        self.universe = 0            # pasangan yang dipindai (setelah filter likuiditas)
        self.deep_candidates = 0     # kandidat analisis mendalam
        self.data_valid = 0          # candle asli cukup & valid
        self.data_incomplete = 0     # None / <60 bar / sintetis / error
        self.score_pass = 0          # skor >= min_score dan uptrend
        self.conflict_reject = 0     # konfluensi sinyal+momentum gagal
        self.signal_confirm = 0      # lolos konfirmasi arah (signal >= thr)
        self.entry_ready = 0         # lolos SEMUA syarat entry
        self.buys_ok = 0
        self.exec_fail = 0
        self.rejections: list[dict] = []
        self._reject_key: dict[str, int] = {}

    # ---- tahap-tahap ----
    def market(self, universe_n: int, src: str):
        self.universe = universe_n
        self._src = src

    def deep_start(self, n: int):
        self.deep_candidates = n

    def data_ok(self):
        self.data_valid += 1

    def data_bad(self, symbol: str, reason: str, detail: str = ""):
        self.data_incomplete += 1
        self.reject(symbol, "market_data", reason, detail)

    def score_pass_one(self):
        self.score_pass += 1

    def signal_confirm_one(self):
        self.signal_confirm += 1

    def entry_ready_one(self):
        self.entry_ready += 1

    # ---- penolakan ----
    def reject(self, symbol: str, stage: str, reason: str,
               actual=None, required=None, extra: str = ""):
        rec = {"symbol": symbol, "stage": stage, "reason": reason}
        if actual is not None:
            rec["actual"] = actual
        if required is not None:
            rec["required"] = required
        if extra:
            rec["extra"] = extra
        self.rejections.append(rec)
        key = f"{stage}:{reason}"
        self._reject_key[key] = self._reject_key.get(key, 0) + 1
        parts = [f"[REJECT] {symbol}", f"stage={stage}"]
        if actual is not None:
            parts.append(f"value={actual}")
        if required is not None:
            parts.append(f"required={required}")
        parts.append(f"reason={reason}")
        if extra:
            parts.append(extra)
        _p(" " + " ".join(str(x) for x in parts))

    # ---- eksekusi ----
    def execution(self, symbol: str, ok: bool, detail: str = ""):
        tag = "[EXEC]"
        if ok:
            self.buys_ok += 1
            _p(f"{tag} BUY {symbol} OK {detail}".rstrip())
        else:
            self.exec_fail += 1
            _p(f"{tag} BUY {symbol} FAILED reason={detail}")

    # ---- cetak ----
    def funnel_lines(self, min_score: float):
        lines = [
            ("dipindai (universe)", self.universe),
            ("analisis mendalam", self.deep_candidates),
            ("data candle valid", self.data_valid),
            ("data tidak lengkap", self.data_incomplete),
            (f"skor >= {min_score:.2f} + uptrend", self.score_pass),
            ("konfirmasi sinyal+momentum", self.signal_confirm),
            ("lolos SEMUA syarat entry", self.entry_ready),
            ("eksekusi BELI berhasil", self.buys_ok),
            ("eksekusi BELI gagal", self.exec_fail),
        ]
        return "\n".join(f"     {name:<34} {n:>4}" for name, n in lines)

    def summary(self, cycle: int, positions_open: int, max_positions: int,
                cash: float):
        dur = time.time() - self.started
        top = sorted(self._reject_key.items(), key=lambda kv: -kv[1])[:3]
        top_s = ", ".join(f"{k} x{v}" for k, v in top) or "-"
        _p(f"[SUMMARY] siklus={cycle} durasi={dur:.1f}s pasar={self.universe} ({getattr(self, '_src', '?')}) "
           f"mendalam={self.deep_candidates} data-valid={self.data_valid} "
           f"ditolak={len(self.rejections)} entry-siap={self.entry_ready} "
           f"trade-baru={self.buys_ok} exec-gagal={self.exec_fail} "
           f"posisi={positions_open}/{max_positions} kas=Rp{cash:,.0f}")
        _p(f"[SUMMARY] alasan-penolakan-teratas: {top_s}")
        _p("[FUNNEL] tahapan kandidat -> transaksi:")
        _p(self.funnel_lines(min_score=getattr(self, "_min_score", 0.62)))
