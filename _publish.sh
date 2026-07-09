#!/usr/bin/env bash
# Render the Quarto site and deploy _site/ to the gh-pages branch.
#
# We deploy from a SEPARATE temp checkout rather than keeping a .git inside
# _site/, because `quarto render` wipes _site/ (and any .git in it) on each run.
# GitHub Pages is configured to serve gh-pages root (build_type: legacy), so no
# `workflow` OAuth scope is needed. To switch to auto-rebuild-on-push instead,
# grant the scope (`gh auth refresh -s workflow`), commit .github/workflows/
# publish.yml, and set Pages source to "GitHub Actions".
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE="https://github.com/LampOfSocrates/trimodal-loss.git"
PUB_DIR="${TMPDIR:-/tmp}/tl_ghpages"

echo "[publish] quarto render"
( cd "$REPO_DIR" && quarto render )

echo "[publish] deploying _site -> gh-pages"
rm -rf "$PUB_DIR"; mkdir -p "$PUB_DIR"
cd "$PUB_DIR"
git init -q && git checkout -q -b gh-pages
git config user.name "LampOfSocrates"
git config user.email "j.sadhukhan@surrey.ac.uk"
cp -r "$REPO_DIR/_site/"* .
touch .nojekyll
git add -A
git commit -q -m "Publish site $(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo local)"
git push -q -f "$REMOTE" gh-pages
echo "[publish] done -> https://lampofsocrates.github.io/trimodal-loss/"
