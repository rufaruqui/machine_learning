# CSE 815 -- Week 1 lecture slides
# Requires: pdflatex, pgfplots, tcolorbox (all in TeX Live)

DECKS := W01_Part1 W01_Part2
PDFS  := $(addsuffix .pdf,$(DECKS))

all: $(PDFS)

%.pdf: %.tex cse815.sty
	pdflatex -interaction=nonstopmode -halt-on-error $<
	pdflatex -interaction=nonstopmode -halt-on-error $<

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fdb_latexmk *.fls

distclean: clean
	rm -f $(PDFS)

.PHONY: all clean distclean
