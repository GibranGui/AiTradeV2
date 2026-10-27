#!/usr/bin/env bash
# Backup state produksi AiTradeV2 sebelum paper run. Tidak menimpa backup sebelumnya:
# setiap eksekusi membuat direktori baru bernama timestamp UTC (mkdir gagal bila sudah ada).
set -euo pipefail

REPO="${REPO:-$HOME/AiTradeV2}"            # ganti jika repo ada di lokasi lain
BACKUP_ROOT="${BACKUP_ROOT:-$HOME/aitrade_backups}"

FILES=(
  "data/paper_wallet.json"
  "data/brain.json"
  "data/experience_buffer.jsonl"
  "ai_trader/data/strategies.json"
)

cd "$REPO"
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "GAGAL: $f tidak ada, backup dibatalkan" >&2; exit 1; }
done

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="$BACKUP_ROOT/$STAMP"
mkdir -p "$BACKUP_ROOT"
mkdir "$DEST"                               # gagal jika sudah ada: tidak pernah menimpa

for f in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  cp -p "$f" "$DEST/$f"
done

# manifest checksum sumber, lalu checksum salinan; keduanya harus identik
( for f in "${FILES[@]}"; do sha256sum "$f"; done ) > "$DEST/SOURCE.sha256"
( cd "$DEST" && for f in "${FILES[@]}"; do sha256sum "$f"; done ) > "$DEST/BACKUP.sha256"
if ! diff -q "$DEST/SOURCE.sha256" "$DEST/BACKUP.sha256" >/dev/null; then
  echo "GAGAL: checksum salinan tidak cocok dengan sumber" >&2; exit 1
fi

echo "Backup OK: $DEST"
cat "$DEST/BACKUP.sha256"
