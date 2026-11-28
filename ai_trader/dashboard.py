"""Dashboard performa AI Trader -> statis HTML (cocok untuk GitHub Pages).

Membaca state bot (data/paper_wallet.json, brain.json, strategies.json,
experience_buffer.jsonl) lalu menghasilkan:
  - Equity curve (modal awal vs ekuitas berjalan per trade selesai)
  - Tabel statistik: total trade, win-rate, PnL, max drawdown, Sharpe
  - Performa per koin (pair)
  - Perkembangan latihan: evolusi strategi & pembelajaran dari kesalahan
  - Posisi terbuka saat ini

CLI:
  python -m ai_trader.dashboard --out site/index.html
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
START_CASH = 10_000_000.0  # modal awal paper trading (IDR)


def _load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def _fmt(v: float) -> str:
    return f"Rp{v:,.0f}"


def build_series(trades: list[dict]) -> tuple[list[dict], dict]:
    """Rekonstruksi kurva ekuitas dari riwayat trade yang sudah selesai (SELL)."""
    sells = [t for t in trades if t.get("action") == "SELL" and t.get("pnl") is not None]
    sells.sort(key=lambda t: t.get("time", 0))
    eq = START_CASH
    peak = eq
    max_dd = 0.0
    rets = []
    series = [{"t": sells[0]["time"] - 1 if sells else 0, "equity": eq}]
    wins = losses = 0
    for s in sells:
        pnl = float(s["pnl"])
        eq += pnl
        rets.append(pnl / max(eq - pnl, 1.0))
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak * 100)
        if pnl >= 0:
            wins += 1
        else:
            losses += 1
        series.append({"t": s["time"], "equity": round(eq, 2)})
    n = len(sells)
    total_pnl = eq - START_CASH
    sharpe = 0.0
    if rets:
        m = sum(rets) / len(rets)
        var = sum((r - m) ** 2 for r in rets) / len(rets)
        sd = math.sqrt(var)
        sharpe = (m / sd) * math.sqrt(min(n, 252)) if sd > 0 else 0.0
    stats = {
        "closed_trades": n,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": (wins / n * 100) if n else 0.0,
        "total_pnl_idr": total_pnl,
        "return_pct": total_pnl / START_CASH * 100,
        "final_equity": eq,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "buys": sum(1 for t in trades if t.get("action") == "BUY"),
    }
    return series, stats


def per_coin_stats(trades: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for t in trades:
        if t.get("action") != "SELL" or t.get("pnl") is None:
            continue
        p = t.get("pair", "?")
        d = out.setdefault(p, {"trades": 0, "wins": 0, "pnl": 0.0})
        d["trades"] += 1
        if t["pnl"] >= 0:
            d["wins"] += 1
        d["pnl"] += float(t["pnl"])
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["pnl"]))


def training_history() -> tuple[list[dict], list[dict]]:
    """Pisahkan log pengalaman menjadi: evolusi strategi & trade selesai."""
    path = os.path.join(DATA_DIR, "experience_buffer.jsonl")
    evos, trades = [], []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("type") == "evolve":
                    evos.append(e)
                else:
                    trades.append(e)
    return evos[-60:], trades[-60:]


HTML_TMPL = """<!doctype html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI TradeV2 — Dashboard</title>
<style>
 :root{{--bg:#0d1117;--card:#161b22;--line:#30363d;--tx:#e6edf3;--dim:#8b949e;
        --up:#3fb950;--down:#f85149;--acc:#58a6ff}}
 *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);
   font:14px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif;padding:24px}
 h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;color:var(--acc);margin:28px 0 10px}
 .sub{color:var(--dim);font-size:12px;margin-bottom:20px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
 .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
 .v{font-size:18px;font-weight:600;margin-top:2px}
 .pos{color:var(--up)}.neg{color:var(--down)}
 svg{background:var(--card);border:1px solid var(--line);border-radius:8px;width:100%;height:auto}
 table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);
   border-radius:8px;overflow:hidden}
 th,td{padding:7px 10px;text-align:left;border-bottom:1px solid var(--line);font-size:13px}
 th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase}
 tr:last-child td{border-bottom:none}
 .note{color:var(--dim);font-size:12px;margin-top:6px}
 .badge{display:inline-block;background:#1f6feb22;color:var(--acc);border:1px solid #1f6feb55;
   border-radius:99px;padding:1px 8px;font-size:11px;margin-left:8px}
</style></head><body>
<h1>🤖 AI TradeV2 — Dashboard Paper Trading <span class="badge">mode simulasi, uang virtual</span></h1>
<div class="sub">Diperbarui otomatis oleh GitHub Actions · @@UPDATED@@ · data pasar asli Indodax/Binance-IDR</div>

<h2>💰 Kinerja</h2>
<div class="grid">@@STATCARDS@@</div>
<div class="note">Modal awal @@MODAL@@ · Ekuitas akhir @@EKUITAS@@
 · Drawdown maks @@DD@@% · Sharpe @@SHARPE@@</div>

<h2>📈 Kurva Ekuitas (per trade selesai)</h2>
@@CHART@@

<h2>🪙 Performa per Koin</h2>
@@COINS@@

<h2>🧠 Perkembangan Latihan AI</h2>
<div class="grid">@@BRAINCARDS@@</div>
@@EVOTABLE@@
<div class="note">Setiap trade rugi → bobot strategi di-penal (credit assignment).
 Tiap siklus evolve → AI mutate parameter, backtest, dan hanya menyimpan varian terbaik.</div>

<h2>📌 Posisi Terbuka Sekarang</h2>
@@OPENTABLE@@

<div class="note" style="margin-top:24px">⚠️ Ini PAPER TRADING dengan harga pasar asli.
 Bukan nasihat keuangan. Jangan aktifkan uang real sebelum ≥100 trade profit konsisten &amp; DD&lt;10%.</div>
</body></html>"""


def make_chart_svg(series: list[dict], w=900, h=280) -> str:
    pts = [(s["t"], s["equity"]) for s in series if s["t"]]
    if len(pts) < 2:
        return '<div class="card">Belum cukup trade selesai untuk menggambar kurva. ' \
               'Grafik akan muncul otomatis setelah Actions mengumpulkan beberapa trade.</div>'
    pad = 40
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys + [START_CASH]), max(ys + [START_CASH])
    if x1 == x0: x1 = x0 + 1
    if y1 == y0: y1 = y0 + 1
    def X(t): return pad + (t - x0) / (x1 - x0) * (w - 2 * pad)
    def Y(v): return h - pad - (v - y0) / (y1 - y0) * (h - 2 * pad)
    poly = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts)
    base = Y(START_CASH)
    up = ys[-1] >= START_CASH
    color = "#3fb950" if up else "#f85149"
    dots = "".join(f'<circle cx="{X(t):.1f}" cy="{Y(v):.1f}" r="2.5" fill="{color}"/>'
                   for t, v in pts[-80:])
    return f'''<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
<line x1="{pad}" y1="{base:.1f}" x2="{w-pad}" y2="{base:.1f}" stroke="#8b949e" stroke-dasharray="4 4"/>
<text x="{pad+4}" y="{base-6:.1f}" fill="#8b949e" font-size="11">modal awal {_fmt(START_CASH)}</text>
<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2"/>{dots}
<text x="{pad}" y="18" fill="#e6edf3" font-size="12">Ekuitas (min {_fmt(y0)} — maks {_fmt(y1)})</text>
</svg>'''


def card(k: str, v: str, cls: str = "") -> str:
    return f'<div class="card"><div class="k">{k}</div><div class="v {cls}">{v}</div></div>'


def render(out_path: str) -> str:
    wallet = _load_json(os.path.join(DATA_DIR, "paper_wallet.json"),
                        {"cash_idr": START_CASH, "positions": [], "trades": []})
    brain = _load_json(os.path.join(DATA_DIR, "brain.json"), {})
    strat = _load_json(os.path.join(os.path.dirname(__file__), "data", "strategies.json"), {})
    trades = wallet.get("trades", [])
    series, stats = build_series(trades)

    stat_cards = "".join([
        card("Trade selesai", f"{stats['closed_trades']}"),
        card("Win rate", f"{stats['win_rate_pct']:.0f}%",
             "pos" if stats["win_rate_pct"] >= 50 else "neg"),
        card("PnL total", _fmt(stats["total_pnl_idr"]),
             "pos" if stats["total_pnl_idr"] >= 0 else "neg"),
        card("Return", f"{stats['return_pct']:+.2f}%",
             "pos" if stats["return_pct"] >= 0 else "neg"),
        card("Kas tersedia", _fmt(wallet.get("cash_idr", 0))),
        card("Posisi terbuka", str(len(wallet.get("positions", [])))),
    ])

    coins = per_coin_stats(trades)
    if coins:
        rows = "".join(
            f"<tr><td>{p}</td><td>{d['trades']}</td><td>{d['wins']}</td>"
            f"<td class='{'pos' if d['pnl']>=0 else 'neg'}'>{_fmt(d['pnl'])}</td></tr>"
            for p, d in coins.items())
        coin_table = ("<table><tr><th>Koin</th><th>Trade</th><th>Menang</th>"
                      f"<th>PnL</th></tr>{rows}</table>")
    else:
        coin_table = '<div class="card">Belum ada trade selesai.</div>'

    evos, exp_trades = training_history()
    if isinstance(strat, dict):
        gens = strat.get("generation", strat.get("generations", len(evos)))
    else:
        gens = len(evos)  # strategies.json berupa daftar strategi aktif
    best_fit = max((e["result"].get("fitness", 0) for e in evos if "result" in e), default=0)
    brain_cards = "".join([
        card("Generasi strategi", str(gens)),
        card("Fitness terbaik", f"{best_fit:.1f}", "pos" if best_fit > 0 else ""),
        card("Entry threshold", str(brain.get("entry_threshold", "-"))),
        card("Stop loss / TP", f"{(brain.get('stop_loss_pct', 0)*100):.1f}% / "
                                f"{(brain.get('take_profit_pct', 0)*100):.1f}%"),
        card("Log pengalaman", f"{len(evos)+len(exp_trades)} entri"),
    ])
    if evos:
        er = "".join(
            f"<tr><td>{datetime.fromtimestamp(e['t'], tz=timezone.utc).strftime('%d/%m %H:%M')}</td>"
            f"<td>{e.get('generations','-')}</td>"
            f"<td>{e['result'].get('trades','-')}</td>"
            f"<td class=\"{'pos' if e['result'].get('return_pct',0)>=0 else 'neg'}\">"
            f"{e['result'].get('return_pct',0):+.2f}%</td>"
            f"<td>{e['result'].get('win_rate',0)*100:.0f}%</td>"
            f"<td>{e['result'].get('fitness',0):.1f}</td></tr>"
            for e in evos[-12:][::-1])
        evo_table = ('<div class="sub">Evolusi strategi terbaru (backtest internal):</div>'
                     "<table><tr><th>Waktu (UTC)</th><th>Gen</th><th>Trades</th>"
                     "<th>Return</th><th>Win%</th><th>Fitness</th></tr>" + er + "</table>")
    else:
        evo_table = '<div class="card">Belum ada log evolusi.</div>'

    positions = wallet.get("positions", [])
    if isinstance(positions, dict):        # format dompet: {pair: {...}}
        positions = list(positions.values())
    if positions:
        pr = "".join(
            f"<tr><td>{p.get('pair','?')}</td><td>{p.get('qty',0):.8f}</td>"
            f"<td>{_fmt(p.get('entry_price',0))}</td>"
            f"<td>{datetime.fromtimestamp(p.get('entry_time',0), tz=timezone.utc).strftime('%d/%m %H:%M')}</td></tr>"
            for p in positions)
        open_table = ("<table><tr><th>Koin</th><th>Jumlah</th><th>Harga masuk</th>"
                      f"<th>Masuk pada (UTC)</th></tr>{pr}</table>")
    else:
        open_table = '<div class="card">Tidak ada posisi terbuka saat ini (bot menahan diri — disiplin, bukan error).</div>'

    updated = datetime.now(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    html = HTML_TMPL
    repl = {
        "@@UPDATED@@": updated,
        "@@STATCARDS@@": stat_cards,
        "@@MODAL@@": _fmt(START_CASH),
        "@@EKUITAS@@": _fmt(stats["final_equity"]),
        "@@DD@@": f"{stats['max_drawdown_pct']:.1f}",
        "@@SHARPE@@": f"{stats['sharpe']:.2f}",
        "@@CHART@@": make_chart_svg(series),
        "@@COINS@@": coin_table,
        "@@BRAINCARDS@@": brain_cards,
        "@@EVOTABLE@@": evo_table,
        "@@OPENTABLE@@": open_table,
    }
    for k, v in repl.items():
        html = html.replace(k, str(v))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render dashboard HTML AI Trader")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "..", "site", "index.html"))
    a = ap.parse_args()
    p = render(a.out)
    print(f"Dashboard ditulis ke {os.path.abspath(p)}")


if __name__ == "__main__":
    main()
