#!/usr/bin/env bash

CRDIR="$(cd "`dirname "$0"`"; pwd)"

nbmerge ?-*.ipynb > main.ipynb

jupyter nbconvert main.ipynb \
 --to slides \
 --ServePostProcessor.reveal_cdn='https://cdnjs.cloudflare.com/ajax/libs/reveal.js/3.7.0/js' \
 --TemplateExporter.exclude_input=True \
 --SlidesExporter.reveal_theme=sky \
 --SlidesExporter.reveal_transition=fade

URI="file://${CRDIR}/main.slides.html?print-pdf#"

echo $URI

chromium-browser --headless --print-to-pdf="main.pdf" "$URI"
