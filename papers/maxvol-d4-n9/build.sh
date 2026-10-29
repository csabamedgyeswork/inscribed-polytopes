#!/usr/bin/env bash
# Render papers/maxvol-d4-n9/main.tex to papers/maxvol-d4-n9/main.pdf.
#
# Uses whichever engine is available, in order of preference:
#   tectonic  - single binary, fetches packages on demand  (brew install tectonic)
#   latexmk   - a full TeX installation                    (brew install --cask mactex-no-gui)
#   docker    - no local TeX needed                        (pulls texlive/texlive)
#
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root/papers/maxvol-d4-n9"

if command -v tectonic >/dev/null 2>&1; then
    echo "engine: tectonic"
    tectonic --keep-intermediates --reruns 3 main.tex
elif command -v latexmk >/dev/null 2>&1; then
    echo "engine: latexmk"
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
elif command -v docker >/dev/null 2>&1; then
    echo "engine: docker (texlive/texlive)"
    docker run --rm -v "$root:/work" -w /work/papers/maxvol-d4-n9 texlive/texlive \
        latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
    echo "no LaTeX engine found; install one of tectonic, latexmk, or docker" >&2
    exit 1
fi

echo
echo "wrote $root/papers/maxvol-d4-n9/main.pdf"
