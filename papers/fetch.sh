#!/bin/sh
# Fetch the primary sources. PDFs are not committed: arXiv's default licence is a
# non-exclusive licence to distribute, granted to arXiv — not to us. See README.md.
set -e
cd "$(dirname "$0")"

fetch() {
    [ -f "$2" ] && { echo "have  $2"; return; }
    echo "fetch $2"
    curl -sSL --max-time 180 -o "$2" "https://arxiv.org/pdf/$1"
}

fetch 1912.04958 1912.04958-stylegan2.pdf
fetch 1812.04948 1812.04948-stylegan.pdf
fetch 1710.10196 1710.10196-progan.pdf
fetch 2006.06676 2006.06676-stylegan2-ada.pdf
fetch 2101.04061 2101.04061-gfpgan.pdf
