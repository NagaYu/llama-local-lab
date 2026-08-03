#!/usr/bin/env bash
#
# Publish Llama Local Lab to a HuggingFace static Space.
#
# Spaces serve a static app from the root of <user>-<space>.static.hf.space, so
# this build must NOT carry the basePath that GitHub Pages needs. That is the
# only difference between the two deploy targets.
#
# Usage:
#   hf auth login          # once — the token is yours, keep it that way
#   ./spaces/deploy.sh [space-id]
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "Not logged in to HuggingFace. Run:  hf auth login" >&2
  exit 1
fi

USER="$(hf auth whoami | head -1 | tr -d '[:space:]')"
SPACE_ID="${1:-${USER}/llama-local-lab}"

echo "▸ Building static export (no basePath, for Spaces)"
unset BASE_PATH
rm -rf .next out
npm run build

echo "▸ Staging upload directory"
rm -rf spaces/build
mkdir -p spaces/build
cp -R out/. spaces/build/
cp spaces/README.md spaces/build/README.md
rm -f spaces/build/.nojekyll   # GitHub Pages marker; meaningless here

if grep -q '/llama-local-lab/_next' spaces/build/index.html; then
  echo "Build carries a basePath — Spaces serves from the root. Unset BASE_PATH." >&2
  exit 1
fi

echo "▸ Ensuring Space $SPACE_ID exists"
hf repo create "$SPACE_ID" --repo-type space --space_sdk static --exist-ok

echo "▸ Uploading"
hf upload "$SPACE_ID" spaces/build . \
  --repo-type space \
  --commit-message "Deploy Llama Local Lab static build"

echo
echo "✓ https://huggingface.co/spaces/$SPACE_ID"
