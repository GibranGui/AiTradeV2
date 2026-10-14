# ROADMAP & Checkpoint Proyek — AiTradeV2

Mode: **PAPER TRADING ONLY** (uang sungguhan dinonaktifkan default; lihat
`ai_trader/live_broker.py` + guard `LIVE_TRADING=1` + `data/LIVE_CONFIRM.txt`).

---

## ✅ 2026-10-14 — Audit "Bot berjalan, 0 transaksi selama 12 jam" (SELESAI)

### Akar masalah (terbukti dari kode + data live, bukan spekulasi)
**Dead-gate logic**: gerbang skor scanner (aditif, ambang 0.62) dan gerbang
konfirmasi entry (`signal >= 0.15` di main.py) tidak pernah disatukan →
kandidat lolos skor gugur senyap di konfirmasi tanpa satu baris log pun.
Bug sekunder: silent failure `PaperBroker.buy()` (return None tanpa alasan),
satu siklus exception mematikan seluruh loop, README/`list` menyesatkan.

Bukti pendukung: state wallet sehat (kas penuh, 0 posisi, MAX_POSITIONS=5
tidak menghalangi); data pasar valid (31 pasar binance-idr; sintetis ditolak
benar); backtest portofolio pada data live menghasilkan trade (event sinyal
valid memang ada — jalur scanner→entry yang tidak menyambungkannya).

### Perbaikan (commit `4f49022`)
| File | Perubahan |
|---|---|
| `ai_trader/scanner.py` | Gerbang terpadu di `opportunities()`: skor ≥ min_score **DAN** uptrend **DAN** signal ≥ 0.15 **DAN** konfluensi momentum; setiap penolakan dicatat per simbol/tahap/nilai/threshold. Try/except per kandidat di `analyze()`. |
| `ai_trader/paper_broker.py` | `last_reject` menyimpan alasan tiap penolakan buy (already_holding_pair / invalid_price / max_positions_reached / insufficient_available_balance). |
| `ai_trader/main.py` | `_scan_cycle()` dipecah + try/except per siklus (kegagalan tercatat ke experience buffer, loop lanjut); log `[ENTRY]`/`[EXEC]`; `do_list()` pakai gerbang sama dengan loop trading. |
| `ai_trader/diag.py` (baru) | Logging terstruktur `[MARKET] [FILTER] [ENTRY] [REJECT] [EXEC] [SUMMARY] [FUNNEL]` — aditif, tidak mengubah keputusan, tidak menyentuh rahasia. |

Yang TIDAK diubah: threshold strategi (0.62 & floor 0.15 tetap), filter risiko,
mekanisme exit (SL/TP/trailing/signal_exit), pembelajaran online + evolusi,
persistensi state, mode PAPER ONLY.

### Pengujian (Fase 5) — suite pytest `tests/test_pipeline.py`
Semua mock/data terkendali di tmp_path; tidak ada transaksi sungguhan.

| # | Tes | Hasil |
|---|---|---|
| 1 | Scanner: data pasar valid (live) | ✅ pass |
| 2 | Scanner: data kosong / exception / sintetis ditolak | ✅ pass (3 tes) |
| 3 | Setiap alasan penolakan entry (score/trend/floor/momentum-conflict) | ✅ pass (4 tes) |
| 4 | Kandidat memenuhi SELURUH syarat entry → lolos | ✅ pass |
| 5 | Saldo tidak mencukupi → `insufficient_available_balance` | ✅ pass |
| 6 | Batas posisi maksimum → `max_positions_reached` | ✅ pass |
| 7 | Pembukaan posisi paper + persisten ke wallet | ✅ pass |
| 8 | Round-trip state + migrasi format wallet lama | ✅ pass |
| 9 | Posisi tertutup tidak dihitung terbuka lagi (sell → slot bebas) | ✅ pass |
| 10 | Satu siklus gagal tidak menghentikan loop (+ tercatat `cycle_error`) | ✅ pass |
| 11 | Entry penuh end-to-end: kandidat valid → `broker.buy` dipanggil → posisi tersimpan | ✅ pass |
| 12 | Konsistensi gerbang: sinyal lemah ditolak DI SCANNER GATE dengan log REJECT | ✅ pass |

**Hasil aktual: `19 passed`** (dijalankan lokal `python -m pytest tests/ -q`;
CI workflow kini menjalankan suite ini otomatis + upload artifact JUnit).

Temuan saat pengujian (bukan bug produksi): desain memakai Series pandas
dengan key posisional → `FutureWarning` deprecation (berkas:
`strategies/portfolio.py`, `scanner.py`) — layak dibersihkan terpisah.

### Diagnosis kondisi saat ini (bukan bug)
Dengan gerbang terpadu, funnel live hari ini menunjukkan pasar IDR-alts
memang belum memberi kandidat yang memenuhi SELURUH syarat (ambang efektif
≈ P(signal>0.25 & skor≥0.62 & uptrend) ≈ langka by design — disiplin, bukan
cacat). Ketika peluang sejati muncul, ia lolos dan dieksekusi dengan log
lengkap. Prioritas proyek: KEBENARAN DIAGNOSIS > KEAMANAN > KUALITAS > FREKUENSI.

---

## ⏭️ Langkah berikutnya
- [ ] Pantau `[SUMMARY]`/`[FUNNEL]` di log Actions beberapa siklus; jika
      `entry-siap>0` tapi `exec-gagal>0`, alasan kini eksplisit di `[EXEC]`.
- [ ] Bersihkan `FutureWarning` indexing Series (`.iloc[i]`) — cosmetic.
- [ ] Alihkan `evolve` dari data sintetis fallback ke data historis asli
      (kualitas evolusi; tidak terbukti menjadi penyebab 0 transaksi).
- [ ] Setelah ≥100 trade selesai & equity curve paper profit konsisten
      (drawdown < 10% selama 1 bulan): uji uang real nominal kecil sesuai
      roadmap keamanan di README.
- [ ] Jika repo diprivatek: Pages butuh alternatif (artifact dashboard).

## 🔐 Catatan keamanan
- Token PAT lama (diawali `ghp_kQ`, tersimpan aman tanpa nilai penuh) terekspos
  WAJIB revoke. Workflow Actions memakai `GITHUB_TOKEN` bawaan + secret
  `BOT_PAT` baru (self-dispatch), tidak membutuhkan token itu.
- API key Indodax real (jika nanti go-live): izin TRADE saja, JANGAN withdraw.
