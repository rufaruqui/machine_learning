# Machine Learning — CSE 815 & CSE 816

Teaching materials for the paired machine learning courses at the **Department of
Computer Science & Engineering, University of Chittagong**:

| Course | Title | Credits | Format |
|---|---|---|---|
| **CSE 815** | Machine Learning (Theory) | 3 | two 1.5-hour lectures per week |
| **CSE 816** | Machine Learning Lab | 1 | four 60-minute lab sessions per week |

CSE 816 is a co-requisite of CSE 815 — the lab implements what the lecture derives, in
the same week. Prerequisites are data structures, algorithms, and one course in
probability and statistics. No linear algebra beyond that is assumed; what the course
needs is built up as it goes.

**Instructor:** Rokan Uddin Faruqui, Professor · <rufaruqui@cu.ac.bd>

Everything here is free to use, adapt, and teach from — see [Licence](#licence).

---

## What is in this repository

```
cse815/     lecture slides    LaTeX/Beamer sources and compiled PDFs
cse816/     lab notebooks     Jupyter notebooks, outputs cleared, run clean top to bottom
book/       the textbook      "Machine Learning: From Linear Models to Reinforcement Learning"
index.html  the course page   a single standalone page linking every released artefact
plan.md     the syllabus      the master plan; everything else renders one week of it
```

`plan.md` is upstream of everything. A change to a topic changes `plan.md`, the deck
that teaches it, the notebook that implements it, the book chapter that explains it,
and the row in `index.html` that links them.

## Syllabus

Three modules of three weeks each, following the arc of a standard first course.

| Module | Weeks | Theory | Lab |
|---|---|---|---|
| **1 — Regression, Classification, Regularization** | 1–3 | linear regression, cost functions, gradient descent, multiple features, vectorization, feature scaling, polynomial regression, logistic regression, overfitting, regularization | model representation, cost and gradient descent from scratch, then with scikit-learn |
| **2 — Neural Networks and ML Practice** | 4–6 | forward propagation, activation functions, softmax, backpropagation, Adam, bias/variance, learning curves, error analysis, skewed data, fairness | networks in NumPy and TensorFlow, multiclass classification, model evaluation |
| **3 — Unsupervised Learning, Recommenders, RL** | 7–9 | *k*-means, anomaly detection, collaborative and content-based filtering, PCA, Markov decision processes, deep Q-learning | clustering, anomaly detection, recommender systems, a deep Q-network |

## Status

| Material | Released |
|---|---|
| Lecture slides | Weeks 1–3 (six decks) |
| Lab notebooks | Weeks 1–2 (eight notebooks) |
| Textbook | Chapters 1–2 written; 22 chapters planned |

The remaining weeks are in preparation. `index.html` shows a grey **⏳ Soon** badge for
anything not yet released, and never links a file that does not exist.

---

## The textbook

`book/` holds a full-length textbook growing out of these notes:

> **Machine Learning: From Linear Models to Reinforcement Learning**

It is deliberately **course-neutral** — no course codes, week numbers, or local
context in the prose — so that it is usable by anyone taking or teaching a first
course in the subject. Twenty-two chapters in nine parts, from linear regression
through neural networks, unsupervised learning and reinforcement learning, to
transformers and large language models, and finally to designing, building and
operating a machine learning system in production.

Two commitments shape it. The reader never meets the phrase *it can be shown that*:
where a result is used, it is derived. And every number printed in it — the value of a
cost function, the output of three iterations of an algorithm, an error rate on held-out
data — is computed by a script in `book/verify/`, never estimated for the sake of a
clean-looking page.

See [`book/BOOK_PLAN.md`](book/BOOK_PLAN.md) for the full chapter map, the design
decisions, and the open questions.

---

## Building and running

### Lecture slides

Requires a TeX distribution with `beamer`, `metropolis`, `pgfplots` and `tcolorbox`,
plus `latexmk` and `gnuplot`. `-shell-escape` is required: cost-surface contours are
produced by pgfplots' gnuplot backend.

```bash
cd cse815
make                 # build every deck
make W02_Part1.pdf   # build one
make check           # clean rebuild, reporting only the warnings that matter
make clean
```

A deck is finished only when `make check` prints `clean: no over/underfull boxes`.

### The textbook

Requires `newpx` (Palatino), `inconsolata`, `tcolorbox`, `biblatex` with `biber`, and the
sans faces `tex-gyre` (TeX Gyre Heros) and `helvetic`. On a minimal TeX Live the last two
are missing and headings silently fall back to a substitute font; install them with

```bash
tlmgr --usermode install tex-gyre helvetic
```

```bash
cd book
make                 # pdflatex → biber → makeindex → pdflatex ×2, via latexmk
make check           # clean rebuild; must report no overfull boxes and no undefined refs
make verify          # recompute every number quoted in the text
```

### Lab notebooks

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r cse816/requirements.txt
jupyter lab
```

Notebooks ship with outputs cleared and must run clean from top to bottom. Every
`assert` in them is a real check that is expected to pass. Each numbered
**Checkpoint** states the value your code should produce, so you can tell whether you
got it right without waiting to be told.

### The course page

`index.html` is a single hand-maintained page with no build step. Serve it rather than
opening it as a `file://` URL:

```bash
python3 -m http.server 8731     # then open http://localhost:8731/index.html
```

---

## Using these materials in your own course

You are welcome to. The slides are plain Beamer built on a single shared style file
(`cse815/cse815.sty`), so recolouring the deck to your own institution is a matter of
editing six colour definitions in one place. The notebooks have no dependencies beyond
NumPy, matplotlib and scikit-learn. The book is written to be readable independently of
any particular course.

If you find an error — in a derivation, a numerical answer, a lab assertion, or a
slide — please open an issue. Corrections to the mathematics are especially welcome,
and a stated expected value that does not match what the code produces is always a bug
worth reporting.

---

## Licence

Copyright © 2026 Rokan Uddin Faruqui.

These materials — the slides, the notebooks, the textbook, and the course page — are
licensed under the **Creative Commons Attribution 4.0 International Licence
(CC BY 4.0)**. You may share and adapt them for any purpose, including commercially,
provided you give appropriate credit, link to the licence, and indicate whether changes
were made. Full terms in [`LICENSE`](LICENSE) and at
<https://creativecommons.org/licenses/by/4.0/>.

Attribution suitable for a slide footer or a syllabus:

> Based on *Machine Learning* course materials by Rokan Uddin Faruqui, Department of
> CSE, University of Chittagong, used under CC BY 4.0.

**Not covered by this licence.** The reference textbooks the course reads from — James
et al., *An Introduction to Statistical Learning*, and Bishop, *Pattern Recognition and
Machine Learning* — are the property of their respective publishers and authors and are
**not** distributed in this repository. Obtain them from their official sources; both
are available free of charge from their authors' pages. The University of Chittagong
crest (`cse815/cu_logo.png`) is a trademark of the University and is included for use
in these materials only; it is not covered by the CC BY grant.

## Citation

```bibtex
@misc{faruqui2026ml,
  author = {Faruqui, Rokan Uddin},
  title  = {Machine Learning: Course Materials for CSE 815 and CSE 816},
  year   = {2026},
  note   = {Department of CSE, University of Chittagong},
  url    = {https://github.com/rufaruqui/machine_learning}
}
```
