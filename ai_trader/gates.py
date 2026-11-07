"""Adaptive Gates — bot memilih & mengendurkan ambangnya sendiri.

Filosofi (permintaan user):
  "Biarkan bot yang memilih dan belajar dari strateginya sendiri, bukan kita
   buatkan template threshold. Kalau diperketat terus, bot tidak pernah
   bertransaksi dan tidak pernah dapat pelajaran — seperti manusia yang takut
   kehilangan Rp50rb untuk satu trade."

Cara kerja:
  * Threshold BUKAN angka mati hasil template manusia — ia adalah GENETIKA
    AI (disimpan di brain.json) yang dioptimalkan lewat backtest evolusioner
    pada data candle ASLI multi-koin. Fitness fungsi memaksa bot berani
    entry: varian dengan NOL trade dihukum (tidak ada trade = tidak ada
    learning), jadi AI secara alami bergerak ke ambang yang menghasilkan
    aktivitas + profitabilitas.
  * Safety-net eksplorasi: jika terlalu lama tidak ada transaksi paper sama
    sekali, threshold diturunkan bertahap (dibatasi batas bawah aman). Ini
    bukan penurunan manual — ini mekanisme belajar "coba dulu, evaluasi
    hasilnya" persis seperti trader pemula yang menambah jam terbang.
  * learn_from_trade tetap menaikkan threshold setelah rugi (belajar dari
    kesalahan); safety-net hanya mengambil alih bila KESENYAPIAN total
    (nol transaksi dalam waktu sangat lama), sehingga dua gaya belajar
    saling menyeimbangkan: ketat setelah rugi, berani setelah lama diam.

Yang TIDAK diubah oleh modul ini: fee, likuiditas, penolakan data sintetis,
maks posisi, mode PAPER-ONLY. Keamanan tetap nomor satu; hanya KEBERANIAN
entry yang menjadi domain AI.
"""

from __future__ import annotations

import time

import numpy as np

# Batas absolut — guardrail keamanan, bukan preferensi strategi.
ENTRY_THR_MIN = 0.10      # sinyal gabungan minimal yang pernah dicoba AI
ENTRY_THR_MAX = 0.45      # tak pernah lebih ketat dari ini (mencekik belajar)
SCORE_THR_MIN = 0.35      # skor peluang tren minimal
SCORE_THR_MAX = 0.62      # nilai template lama = batas atas (warisan)
SCALP_STRONG_MIN = 0.10
SCALP_FLOOR_MIN = 0.04

# Safety-net kesunyian: setelah X menit tanpa SATU pun trade paper (buy/sell),
# turunkan keberanian-sedikit demi sedikit sampai Y minimum.
SILENCE_DECAY_AFTER_SEC = 4 * 3600     # 4 jam tanpa aktivitas -> mulai turun
SILENCE_DECAY_STEP_SEC = 2 * 3600      # tiap 2 jam sunyi tambahan, turun lagi
SILENCE_STEP_SIZE = 0.02               # penurunan entry_threshold per langkah


def default_gates() -> dict:
    """Nilai AWAL (bukan final) — selanjutnya AI yang menyesuaikan."""
    return {
        "entry_threshold": 0.20,
        "score_min": 0.45,
        "signal_floor": 0.08,
        "scalp_strong": 0.15,
        "scalp_floor": 0.06,
    }


def sync_brain(brain: dict) -> dict:
    """Pastikan otak punya kunci gerbang adaptif + batas-batasnya."""
    d = default_gates()
    for k, v in d.items():
        brain.setdefault(k, v)
    # clamp terhadap batas absolut (jaga dari state korup/manual-edit liar)
    brain["entry_threshold"] = float(np.clip(brain["entry_threshold"],
                                             ENTRY_THR_MIN, ENTRY_THR_MAX))
    brain["score_min"] = float(np.clip(brain.get("score_min", d["score_min"]),
                                       SCORE_THR_MIN, SCORE_THR_MAX))
    brain["signal_floor"] = max(SILENCE_STEP_SIZE, float(
        brain.get("signal_floor", d["signal_floor"])))
    return brain


def _last_activity_ts(broker) -> float:
    ts = [t.get("time", 0) for t in broker.trades] or [0]
    ts += [p.entry_time for p in broker.positions.values()]
    return max(float(x) for x in ts)


def apply_silence_decay(brain: dict, strategies, broker, now: float | None = None) -> list[str]:
    """Safety-net anti-'takut-rugi': kalau sudah sangat lama tanpa ENTRY BARU,
    AI menurunkan ambangnya sendiri setahap demi setahap (sampai batas bawah).

    FIX "reject jauh lebih banyak daripada open": sebelumnya safety-net ini
    hanya aktif saat sunyi TOTAL (nol trade & nol posisi). Akibatnya satu
    posisi yang menggantung lama membekukan jangkar kesunyian selamanya ->
    gerbang tidak pernah melonggar -> bot terus-menerus reject tanpa pernah
    menambah pengalaman. Sekarang yang dihitung adalah KEBERANIAN (entry
    terakhir): selama tidak ada posisi/baru dibuka, AI berlatih memberanikan
    diri — meski masih ada posisi lama yang belum selesai."""
    now = now or time.time()
    lessons: list[str] = []
    last_entry = max([float(p.entry_time or 0) for p in broker.positions.values()]
                     + [float(t.get("time", 0) or 0) for t in broker.trades]
                     + [0.0])
    if last_entry and now - last_entry < SILENCE_DECAY_AFTER_SEC:
        brain["_silence_anchor"] = now       # baru saja entry -> reset jangkar
        return lessons
    anchor = float(brain.get("_silence_anchor") or now)
    idle = now - anchor
    if idle < SILENCE_DECAY_AFTER_SEC:
        return lessons
    steps = int((idle - SILENCE_DECAY_AFTER_SEC) // SILENCE_DECAY_STEP_SEC) + 1
    changed = False
    for _ in range(min(steps, 12)):
        before = (brain["entry_threshold"], brain["score_min"],
                  brain.get("signal_floor", 0.08), brain.get("scalp_strong", 0.15),
                  brain.get("scalp_floor", 0.06))
        brain["entry_threshold"] = round(max(ENTRY_THR_MIN,
                                   brain["entry_threshold"] - SILENCE_STEP_SIZE), 3)
        brain["score_min"] = round(max(SCORE_THR_MIN,
                             brain.get("score_min", 0.45) - 0.02), 3)
        brain["signal_floor"] = round(max(0.0,
                                brain.get("signal_floor", 0.08) - 0.01), 3)
        brain["scalp_strong"] = round(max(SCALP_STRONG_MIN,
                                brain.get("scalp_strong", 0.15) - 0.01), 3)
        brain["scalp_floor"] = round(max(SCALP_FLOOR_MIN,
                               brain.get("scalp_floor", 0.06) - 0.01), 3)
        after = (brain["entry_threshold"], brain["score_min"],
                 brain.get("signal_floor", 0.08), brain.get("scalp_strong", 0.15),
                 brain.get("scalp_floor", 0.06))
        if after != before:
            changed = True
    if changed:
        brain["_silence_anchor"] = now       # kasih waktu untuk membuktikan diri
        lessons.append(
            f"AI melonggarkan gerbang sendiri (idle {idle / 3600:.1f}h tanpa trade): "
            f"entry_thr->{brain['entry_threshold']:.2f} score_min->{brain['score_min']:.2f} "
            f"floor->{brain['signal_floor']:.2f} scalp->{brain['scalp_strong']:.2f}/"
            f"{brain['scalp_floor']:.2f} "
            "(belajar butuh pengalaman; batas bawah aman tetap dijaga)")
    return lessons


def gate_snapshot(brain: dict) -> dict:
    return {k: brain[k] for k in
            ("entry_threshold", "score_min", "signal_floor",
             "scalp_strong", "scalp_floor") if k in brain}
