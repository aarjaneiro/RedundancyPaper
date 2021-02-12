#!/bin/bash
read -p "Makeindex? Default: N" MAKEINDEX
pdflatex -interaction nonstopmode main # Run for bibtex init
bibtex main
if [ MAKEINDEX == "Y" ]; then
  makeindex
fi
pdflatex -interaction nonstopmode main # Pass twice
pdflatex -interaction nonstopmode main
echo "Moving to output"
cp main.pdf main.bib output
