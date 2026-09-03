# Makefile for the Machine Learning textbook.
#
#   make          build main.pdf (latexmk drives pdflatex/biber/makeindex)
#   make check    clean rebuild, then print only the warnings that matter
#   make clean    remove auxiliary files
#   make distclean  also remove main.pdf
#
# -shell-escape is required: cost-surface contours use pgfplots' gnuplot
# backend, which shells out.

LATEXMK  = latexmk
FLAGS    = -pdf -shell-escape -interaction=nonstopmode -halt-on-error
MAIN     = main

.PHONY: all check clean distclean verify

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex mlbook.sty bibliography.bib \
             $(wildcard ch*.tex) $(wildcard appendix*.tex) \
             preface.tex notation.tex
	$(LATEXMK) $(FLAGS) $(MAIN).tex

# A chapter is done only when this prints "clean".
check: distclean all
	@echo "---------------- warnings that matter ----------------"
	@grep -E "Overfull \\\\hbox \(([5-9]|[1-9][0-9]+)\..*pt" $(MAIN).log \
	   || echo "boxes: no overfull hbox over 5pt"
	@grep -E "Overfull \\\\vbox" $(MAIN).log || echo "boxes: no overfull vbox"
	@grep -E "(There were undefined references|Citation .* undefined|Reference .* undefined)" \
	   $(MAIN).log || echo "refs: all references and citations resolved"
	@echo "------------------------------------------------------"

# Regenerate every number quoted in the book.
verify:
	@PY=$${PY:-/tmp/mlenv/bin/python}; $$PY verify/run_all.py

clean:
	$(LATEXMK) -c $(MAIN).tex
	rm -f $(MAIN).idx $(MAIN).ind $(MAIN).ilg $(MAIN).bbl $(MAIN).run.xml \
	      $(MAIN).aux *.gnuplot *.table *_contourtmp* *.bbl-SAVE-ERROR

distclean: clean
	rm -f $(MAIN).pdf
