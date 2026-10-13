# 🤖 AI Self-Learning Trader — Paper Trading dengan Data Asli Indodax

AI trading yang **belajar dari kesalahannya sendiri**, menggabungkan tiga keluarga
strategi (**indikator teknikal, bentuk candlestick, analisis volume**), mampu
**memodifikasi strateginya sendiri** (self-evolving configuration), dan terhubung
ke **Indodax** untuk **paper trading dengan harga pasar asli** (tanpa uang sungguhan).

## Arsitektur

```
ai_trader/
├── market_data.py         # Konektor API publik Indodax (ticker, trades, OHLCV) + fallback offline
├── features.py            # Fitur: RSI/MACD/EMA/Bollinger/Stochastic/ATR + 10 pola candle + volume (OBV, VWAP, spike)
├── strategies/portfolio.py# 8 strategi modular, bobot & parameter, mutasi (self-modding), persist JSON
├── paper_broker.py        # Simulasi eksekusi di harga ASLI Indodax, fee 0,3%, dompet virtual Rp10 juta
├── learning.py            # Otak AI: credit assignment, adaptive risk, evolusi backtest
└── main.py                # Loop real-time + CLI
data/                      # State yang dipelajari (wallet, brain.json, strategies.json, experience_buffer.jsonl)
requirements.txt           # Dependensi (requests, pandas, numpy)
.github/workflows/
├── paper-trader.yml       # GitHub Actions: jalankan loop bot otomatis tiap 30 menit
└── ci.yml                 # Smoke-test setiap push/PR
```

## Quick Start (30 detik)

```bash
git clone https://github.com/GibranGui/AiTradeV2.git
cd AiTradeV2
pip install -r requirements.txt
python -m ai_trader.main evolve --pair IDRBTC --interval 60   # AI belajar dari data historis
python -m ai_trader.main run --pair IDRBTC --poll 30           # paper trading live
python -m ai_trader.main status                                # lihat dompet & pelajaran
```

## Cara AI Belajar dari Kesalahan

1. **Credit assignment per trade** — setiap posisi dibuka, snapshot fitur disimpan.
   Saat ditutup rugi, strategi yang dominan mendorong entry diturunkan bobotnya;
   saat profit, dinaikkan (`learning.learn_from_trade`).
2. **Adaptive risk control** — kerugian besar memperketat stop-loss; loss beruntun
   menaikkan ambang entry (lebih selektif); win-rate rendah mengecilkan ukuran posisi.
3. **Evolusi mandiri (self-modifying)** — AI membuat varian portofolio baru lewat
   `mutate()` (termasuk melahirkan strategi baru dari template), meng-backtest-nya
   pada data historis Indodax, dan hanya menyimpan konfigurasi terbaik ke
   `strategies.json`. Ini cara aman "mengubah script sendiri": AI menulis ulang
   *kode strategi dalam bentuk konfigurasi tervalidasi*, bukan men-edit file .py
   sembarangan (bebas risiko crash/injeksi).
4. **Experience buffer** — semua pelajaran dicatat di `data/experience_buffer.jsonl`
   dan bisa diaudit.

## Menjalankan

```bash
pip install requests pandas numpy

# 1) Biarkan AI belajar/berevolusi dari data historis Indodax dulu (disarankan)
python -m ai_trader.main evolve --pair IDRBTC --interval 60

# 2) Jalankan paper trader live (harga asli Indodax, refresh tiap 30 dtk)
python -m ai_trader.main run --pair IDRBTC --interval 60 --poll 30 --evolve-every 30

# 3) MODE MULTI-KOIN (disarankan): scan SELURUH pasar IDR tiap siklus,
#    auto-trade koin yang terdeteksi peluang tren naik (maks 5 posisi).
python -m ai_trader.main scan --interval 60 --poll 45 --deep 12 --min-vol 50000000

# 4) Lihat daftar peluang saat ini tanpa trading (one-shot)
python -m ai_trader.main list --deep 12 --min-vol 50000000

# 5) Cek dompet, otak (aturan risiko), bobot strategi, dan riwayat pelajaran
python -m ai_trader.main status
```

Bot berjalan terus (Ctrl+C untuk berhenti; state tersimpan otomatis).
Jika tidak ada internet, sistem fallback ke data sintetis agar tetap bisa diuji.

## Altcoin Scanner (multi-koin, bukan satu koin saja)

Terinspirasi pola pipeline *gstack* (`sweep wide → filter → rank → act`), mode `scan`:

1. **Sweep** — satu request mengambil ticker 24 jam SEMUA pasar berpasangan IDR
   (mirror resmi Binance-IDR / Indodax; ±58 pasar non-stablecoin).
2. **Filter** — buang stablecoin & koin tak likuid (`--min-vol`, default Rp50 juta/24 jam);
   pertahankan yang sedang naik ATAU baru bangkit (harga dekat high 24 jam).
3. **Rank** — analisis mendalam candle OHLCV asli per kandidat (indikator + pola
   candlestick + volume, memakai bobot strategi terpelajari AI).
4. **Act** — beli maksimal 5 koin dengan skor tren naik tertinggi; TP/SL/trailing
   dievaluasi tiap siklus untuk SEMUA posisi; setiap trade (untung/rugi) masuk
   ke loop pembelajaran.

## Jalankan 24/7 dengan GitHub Actions (gratis, tanpa server)

Repo ini punya workflow [`.github/workflows/paper-trader.yml`](.github/workflows/paper-trader.yml)
yang menjalankan bot di runner GitHub — **tidak perlu VPS**.

- **Otomatis**: cron tiap 30 menit → restore state → jalankan loop ±25 menit → simpan state.
- **Manual**: tab *Actions* → *Paper Trader (live loop)* → *Run workflow* → isi `pair`, `interval`, `poll`, `evolve_every`.
- **State persisten** (dompet, otak, strategi, pengalaman) disimpan di branch terpisah
  **`bot-state`**, plus laporan singkat di `bot-report.txt`. Kode di `main` tidak pernah tercemar.
- CI tambahan (`.github/workflows/ci.yml`) melakukan smoke-test (`evolve` + `status`) setiap push/PR.

> ⚠️ Aktifkan **Settings → Actions → General → Workflow permissions → Read and write
> permissions** agar workflow bisa push ke branch `bot-state` (memakai `GITHUB_TOKEN`,
> bukan token pribadimu).

## Catatan Penting

- **Paper trading**: order dieksekusi simulasi pada *last price* ticker asli
  Indodax (public API, tanpa kunci API). Tidak ada dana riil.
- Untuk exchange live lain atau akun riil, perlu private API + tanda tangan HMAC —
  sengaja TIDAK diaktifkan demi keamanan.
- Backtest/evolusi memakai data historis terbatas (≤600 candle) → hasil adalah
  latihan adaptasi, bukan jaminan profit. Crypto sangat volatil; gunakan ini untuk
  edukasi.
