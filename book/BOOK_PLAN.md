# Machine Learning: From Linear Models to Reinforcement Learning

**Author:** Rokan Uddin Faruqui
**Audience:** Undergraduates taking a first machine learning course anywhere, and
self-learners with the same background. Assumes data structures, algorithms, introductory
probability and statistics, and working Python. Assumes **no** linear algebra beyond a first
statistics course — the matrix calculus the book needs is built up in Appendix A.
**Format:** Professional B5 textbook, ~600–700 pages, pdfLaTeX
**Design goal:** A student who reads this book and works its exercises can pass a first ML course
without attending a lecture, and can then build, ship, and operate a model.

## The book is course-neutral

The text names no university, course code, semester, or currency. It is a textbook that happens to
have grown out of a specific course, not the course written down. Concretely:

- No `CSE 815` / `CSE 816`, no "Week 3", no "this semester" anywhere in the prose.
- Practical work appears as an **In the Lab** box naming a numbered lab ("Lab 3.2 — Gradient
  descent from scratch"), never a notebook path. The numbered labs are shipped alongside the book;
  which local course week they fall in is not the book's business.
- Examples use internationally legible units: floor area in hundreds of square feet, prices in
  thousands of dollars, temperatures in Celsius. Case studies name a role ("a hospital triage
  team", "a regional utility"), not a country.
- The author's affiliation stays on the title page. That is a byline, not a course tag.

A mapping from chapters to the local 9-week delivery lives at the bottom of this file, **outside**
the book, so the two can move independently.

---

## File Structure

```
book/
  main.tex            ← master file (\input all chapters)
  mlbook.sty          ← shared style: packages, palette, tcolorbox defs, math macros, listings
  preface.tex         ← preface + how to use this book
  notation.tex        ← the notation table, front matter
  ch01.tex … ch21.tex
  appendixA.tex … appendixE.tex
  bibliography.bib    ← ALL entries, BibLaTeX/biber format
  verify/             ← Python scripts that COMPUTE every number printed in the book
  Makefile
  BOOK_PLAN.md        ← this file
```

Note the deviation from `../data_engineering/book/`, which uses a bare `preamble.tex`. This repo's
convention is a shared `.sty`, so the book follows the repo, not the sibling.

Build (`make`, which drives `latexmk`):

```
pdflatex -shell-escape main.tex ; biber main ; makeindex main ; pdflatex ×2
```

`-shell-escape` is mandatory for the same reason as the slides: cost-surface contours come from
pgfplots' gnuplot backend. `make check` does a clean rebuild and prints only the warnings that
matter; a chapter is not done until it reports no overfull box over 5 pt and no undefined
reference or citation.

---

## LaTeX Design Decisions

| Element | Choice |
|---|---|
| Document class | `\documentclass[11pt,b5paper,twoside,openright]{book}` |
| Text font | `newpxtext` (Palatino clone, pdfLaTeX) |
| Math font | `newpxmath` — chosen for dense display math; ML pages are half equations |
| Mono font | `inconsolata` |
| Typography | `microtype` (`expansion=false`), `\setstretch{1.15}` |
| Page layout | `geometry`: top 2.5 cm, bottom 2.8 cm, inner 2.8 cm, outer 2.2 cm |
| Colors | The lecture-slide palette, so book and deck are visibly one body of work |
| Chapter heads | `titlesec`: oversized grey chapter number + MLBlue rule |
| Headers | `fancyhdr`: chapter name verso, section name recto, folio outside |
| Bibliography | `biblatex` author-year + `biber` |
| Code | `listings`: styles `pythonml`, `tfkeras`, `pseudocode`, `shell` |
| Figures | Hand-written TikZ/pgfplots only — no raster images |
| Boxes | `tcolorbox` — 10 environments (below) |
| Cross-refs | `hyperref` + `cleveref`; `makeidx` index |

### Palette

| Macro | Hex | Role |
|---|---|---|
| `MLBlue` | `#123B63` | structure, chapter numbers, definitions |
| `MLTeal` | `#1B7F79` | key insights, worked examples |
| `MLAmber` | `#C97B17` | derivations, algorithms |
| `MLRose` | `#9B2C4B` | pitfalls |
| `MLGray` | `#4A5560` | lab, ethics and production notes, folios |
| `MLLight` | `#EEF3F7` | box backgrounds |

### tcolorbox Environments

| Environment | Color | Purpose |
|---|---|---|
| `chapterlearning` | MLBlue fill | Learning objectives, opens every chapter |
| `defbox` | white + MLBlue border | Formal definition |
| `keyinsight` | MLTeal | The one idea to carry out of the section |
| `pitfall` | MLRose | Misconception / exam trap |
| `derivation` | MLAmber, breakable | Line-by-line algebra, every step justified |
| `workedexample` | MLTeal tint, breakable | Small numeric example carried through by hand |
| `algorithmbox` | MLAmber tint | Boxed pseudocode |
| `researchspotlight` | MLBlue | One-page summary of a landmark paper |
| `labbox` | MLGray | "Do this now": the numbered lab, its checkpoint, its expected value |
| `ethicsnote` | MLGray | Fairness, deployment, and misuse — a thread, not one chapter |

The first four mirror the beamer boxes deliberately: a student who saw a *Key idea* on a slide
meets the same box, same colour, in the book.

### Notation

`notation.tex` fixes notation once, and it is the slide notation: `\xii{i}`, `\yii{i}`,
`\xjj{i}{j}`, `\fwb`, `\fvwb`, `\vw`, `\vx`, `\yhat`, `\argmin`, `\argmax` — same macro names as
the deck style file, so text moves between the two without retyping. Additions the book needs and
the slides do not: `\Xmat`, `\loss` (per-example loss, vs. `J` for cost), `\grad`, `\Exp`,
`\Prob`, `\Real{n}`, `\gsig`, `\aL{l}{j}`, `\Wl{l}`, `\bl{l}`, `\Hspace` (hypothesis space).

A two-page notation table in the front matter lists every symbol with its first-use chapter.

### A note on the page budgets

The per-chapter page counts below were estimates made before any chapter existed. The six
written so far came in at 24, 22, 20, 18, 22 and 31 B5 pages against estimates of 30, 34, 38, 36,
34 and 40 — roughly **63% of estimate**. Chapters 1–5 settled near 21 pages once the template
stabilised; Ch 6 is the first materially longer one, because it rebuilds the model, the cost and
the decision rule in one chapter.
The remaining estimates have *not* been revised downward individually, since the later chapters
(neural networks, transformers, the systems part) are genuinely denser; but the whole-book figure
above has been, from ~850 pages to ~650. Treat a chapter's number as an upper bound, not a target:
a chapter is done when its outline is covered and its exercise quotas are met, not when it reaches
a page count.

### Exercise Numbering Scheme

- **SQ** N.k — Short Question (definitions, one-line reasoning) — 8–12 per chapter
- **AP** N.k — Analytical Problem (derive, prove, compute by hand) — 5–8 per chapter
- **DP** N.k — Design Problem (choose a model/metric/pipeline, justify) — 3–5 per chapter
- **PE** N.k — Programming Exercise (NumPy first, library second) — 3–5 per chapter
- **QZ** N.k — Multiple-choice item for self-testing — 6–10 per chapter

Counters reset per chapter. Appendix E gives worked answers to roughly one third of SQ/AP and
hints for the rest; DP answers are rubrics, not solutions. SQ + AP are the descriptive half of
every chapter's assessment, DP the design half.

### Reading policy

Every chapter closes with a "Further Reading" paragraph naming, with section numbers,

1. James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning*, 2e — **primary**
2. Bishop, *Pattern Recognition and Machine Learning* — **supplementary**, only where ISLR is thin

and nothing else. Four chapters and one part fall outside ISLR/Bishop coverage and name one
substitute each, replacing rather than adding to the two-book budget: Sutton & Barto for Ch 17,
Jurafsky & Martin *Speech and Language Processing* 3e for Ch 18 and Ch 19, and Huyen *Designing
Machine Learning Systems* for Part IX. Landmark papers live in `researchspotlight` boxes and the bibliography, not in the
reading list, so exam preparation stays two books wide.

---

## Part and Chapter Map

| Part | Ch | Title | Pages | Status |
|---|---|---|---|---|
| **I — Foundations and Linear Regression** | 1 | What Machine Learning Is | 24 | **DONE** |
| | 2 | Linear Regression with One Variable | 22 | **DONE** |
| | 3 | Gradient Descent | 20 | **DONE** |
| | 4 | Multiple Linear Regression and Vectorization | 18 | **DONE** |
| | 5 | Making Gradient Descent Work | 22 | **DONE** |
| **II — Classification and Generalization** | 6 | Logistic Regression | 40 | **DONE** (31) |
| | 7 | Overfitting and Regularization | 36 | **DONE** (26) |
| **III — Neural Networks** | 8 | Neural Networks: Representation and Forward Propagation | 40 | **DONE** (22) |
| | 9 | Training Neural Networks | 44 | **DONE** (28) |
| **IV — Making Learning Systems Work** | 10 | Evaluation, Bias and Variance | 38 | **DONE** (22) |
| | 11 | Error Analysis, Data, Skew, and Ethics | 36 | **DONE** (20) |
| | 12 | Decision Trees and Ensembles | 34 | **DONE** (22) |
| **V — Unsupervised Learning** | 13 | Clustering with *k*-Means | 30 | **DONE** (14) |
| | 14 | Anomaly Detection | 30 | **DONE** (18) |
| **VI — Recommenders and Dimensionality Reduction** | 15 | Recommender Systems | 40 | PLANNED |
| | 16 | Principal Component Analysis | 26 | PLANNED |
| **VII — Reinforcement Learning** | 17 | Reinforcement Learning and Deep Q-Networks | 42 | PLANNED |
| **VIII — Transformers and Language Models** | 18 | Sequence Models and the Transformer | 36 | PLANNED |
| | 19 | Large Language Models in Practice | 34 | PLANNED |
| **IX — Machine Learning Systems** | 20 | Designing a Machine Learning System | 38 | PLANNED |
| | 21 | The Development Pipeline: Data, Features, Training, Evaluation | 40 | PLANNED |
| | 22 | Deployment, Monitoring, and the Operating Model | 42 | PLANNED |

### Back matter
| # | Title | Pages |
|---|---|---|
| A | Mathematical Background: Vectors, Matrices, Derivatives, Probability | 34 |
| B | Python, NumPy, scikit-learn and TensorFlow for This Course | 24 |
| C | Notation, Symbols, and Glossary | 12 |
| D | Datasets Used in the Book and the Labs | 8 |
| E | Hints and Selected Solutions | 30 |

Front matter: title/copyright (CC BY 4.0), preface, **How to use this book** (three reading paths:
a one-semester core, a shorter exam-revision path, and a practitioner path that jumps from Ch 7
to Part IX), notation table, contents.

**Chapters track lecture-sized units**: Ch 1–17 are one 1.5-hour session each, which is what makes
the book usable as assigned reading. Ch 18–22 are the exception — they are written as reading
chapters, sized for a two-session pair or for self-study, because the material has no fixed
lecture shape.

---

## Chapter Outlines

Every chapter follows one template: epigraph → `chapterlearning` → motivating problem → numbered
sections → chapter summary → exercises (SQ/AP/DP/PE/QZ) → Further Reading.

### Ch 1 — What Machine Learning Is  *(done)*
Classical programming vs. learning from data · where ML is used and the common thread · Samuel's
and Mitchell's definitions, and instantiating ⟨*T*, *P*, *E*⟩ · the four parts of every supervised
learner: data, hypothesis space, cost, optimizer · supervised learning: regression and
classification · unsupervised learning: clustering, anomaly detection, dimensionality reduction ·
where reinforcement learning sits · a first end-to-end example, fitted, evaluated against a
baseline, and tested on held-out data · what ML is *not*: known rules, causality, data quality ·
a map of the book.
Boxes: `defbox` ×4; `keyinsight` ×3; `pitfall` ×3; `workedexample` (the end-to-end fit);
`ethicsnote`; `labbox`; `researchspotlight` Samuel (1959), Mitchell (1997).
Reading: ISLR §1, §2.1–2.2; Bishop §1.1.

### Ch 2 — Linear Regression with One Variable  *(done)*
The model and what each parameter does · training-set notation · squared-error cost, and why the
½ and the 1/*m* are there · `J(w)` with `b` fixed: the parabola · `J(w,b)`: the bowl, contour
plots, and how to read one · least squares as projection, geometrically · fitting a 4-point set by
hand.
Boxes: `derivation` of `J` from the residual vector; `workedexample` `J` at three parameter pairs;
`pitfall` "minimising `J` is not the same as a good model".
Reading: ISLR §3.1; Bishop §1.1, §3.1.1.

### Ch 3 — Gradient Descent  *(done)*
The descent idea and the update rule · simultaneous update, and the bug you get without it ·
derivative intuition in one dimension · the learning rate: too small, too large, and the
divergence threshold **derived** as `2/λ_max` for a quadratic cost, not asserted · gradient descent
for linear regression, both partials derived in full · convexity of the squared-error cost · a
hand-computed iteration table, then the same table from code · batch vs. stochastic vs. mini-batch.
Boxes: `derivation` ×2; `algorithmbox`; `workedexample` three iterations by hand; `pitfall`
non-simultaneous update; `researchspotlight` Robbins & Monro (1951).
Reading: ISLR §3.1; Bishop §3.1.3, §5.2.4.

### Ch 4 — Multiple Linear Regression and Vectorization  *(done)*
From one feature to *n* · the design matrix · model in vector form · vectorization: the loop, the
dot product, and the measured speed-up · gradient descent for *n* features in vector form · the
normal equation, derived, with its `O(n³)` cost and its singular cases · multicollinearity.
Boxes: `derivation` ×2; `workedexample` a 2-feature fit both ways agreeing to six decimals;
`keyinsight` "vectorization changes the implementation, not the algorithm"; `pitfall` inverting
`XᵀX` with duplicated features.
Reading: ISLR §3.2; Bishop §3.1.

### Ch 5 — Making Gradient Descent Work  *(done)*
Feature scaling and the elongated-contour picture · convergence checks, and how each can lie ·
choosing the step size by the ×3 ladder · feature engineering as encoded domain knowledge ·
polynomial regression as linear regression on transformed features · a first informal sighting of
overfitting.
Boxes: `workedexample` scaling changing the usable step size by an order of magnitude; `pitfall`
scaling the test set with test-set statistics; `algorithmbox` z-score scaling; `keyinsight` "linear
in the *parameters*, not the features".
Reading: ISLR §3.3, §7.1; Bishop §1.1, §3.1.

### Ch 6 — Logistic Regression  *(done)*
Why squared error fails for classification: the non-convex cost, drawn · the sigmoid and its
derivative, derived (needed again in Ch 9) · output as a probability · decision boundaries, linear
then polynomial · the logistic loss from maximum likelihood, derived in full · the simplified
one-line loss · convexity · gradient descent, and why the update has the same *form* as linear
regression · thresholds other than 0.5.
Boxes: `derivation` ×4; `workedexample` ×3, including two gradient steps by hand; `pitfall` ×4,
one of them "same update rule ⇒ same algorithm"; `researchspotlight` Cox (1958), Rosenblatt
(1958); `ethicsnote` on who bears the false negatives.
Reading: ISLR §4.1–4.3; Bishop §4.2, §4.3.2–4.3.3.

**Deviation from this outline, as written.** "Why squared error fails" is split in two, because the
non-convexity argument cannot be made before the sigmoid model exists. §6.2 gives the failure of
*linear regression* on a binary target (out-of-range outputs; one distant, correctly-classified
example dragging the boundary from 4.50 to 5.57), and §6.5 the failure of *squared error on the
sigmoid model* (negative Hessian eigenvalues; a vanishing gradient in the tails). The chapter file
records this in a header comment.

New datasets, both defined in the chapter and belonging in Appendix D: **T7**, eight tumours by
diameter with one label inversion — chosen so that Σy = m/2 makes the first gradient step
hand-computable and the fitted boundary lands on exactly 4.5 cm; **T8**, twenty tumours by size
and texture, with genuine class overlap so the fit stays finite.

### Ch 7 — Overfitting and Regularization
The three-panel under/just-right/over figure with matching numbers · why more parameters buy
training accuracy and can cost test accuracy · the three remedies · the regularized cost and what
λ trades off · regularized linear regression, with `(1 − αλ/m)` read as shrinkage · regularized
logistic regression · why `b` is not regularized · L2 vs. L1 through the constraint-region picture
· choosing λ.
Boxes: `derivation` shrinkage factor; `workedexample` a λ sweep on a degree-9 fit with a
train/test table; `pitfall` regularizing unscaled features; `researchspotlight` Hoerl & Kennard
(1970), Tibshirani (1996).
Reading: ISLR §6.2, §2.2.2; Bishop §1.1, §3.2.

### Ch 8 — Neural Networks: Representation and Forward Propagation
Biological inspiration and where it stops being useful · from features to learned features · a
layer as many logistic units, in vector form · layer notation · forward propagation, layer by
layer then as one matrix product · why linear activations collapse the network — proved in two
lines · what early layers learn · the same network in three lines of Keras, with the shape
conventions that trip students up · a NumPy implementation matching it.
Boxes: `derivation` linear-activation collapse; `algorithmbox` forward propagation;
`workedexample` a 2-2-1 network computed fully by hand; `pitfall` row- vs. column-vector shapes;
`researchspotlight` McCulloch & Pitts (1943), Cybenko (1989).
Reading: ISLR §10.1–10.2; Bishop §5.1.

### Ch 9 — Training Neural Networks
Model, loss, fit — and what each line actually runs · activation functions and how to choose per
layer · why a nonlinearity is required · softmax, its cost, and its gradient · numerically stable
softmax and the overflow it prevents · multiclass vs. multi-label · backpropagation: computation
graphs, the chain rule, and a full worked backward pass · Adam and per-parameter step sizes ·
convolutional layers, briefly, as the bridge to Ch 18.
Boxes: `derivation` backprop for one hidden layer, and the softmax + cross-entropy simplification;
`algorithmbox` backpropagation; `workedexample` one full forward+backward pass with numbers;
`pitfall` the softmax round-off bug; `researchspotlight` Rumelhart, Hinton & Williams (1986),
Nair & Hinton (2010), Kingma & Ba (2015), LeCun et al. (1998).
Reading: ISLR §10.7; Bishop §5.2–5.3.

### Ch 10 — Evaluation, Bias and Variance
The diagnostic mindset · train/test, and why a test set touched twice is not a test set ·
train/validation/test and model selection · the bias/variance signature table · regularization and
bias/variance · baselines, and why an error rate without one is not a diagnosis · learning curves,
and what they say about collecting more data · bias/variance in neural networks · *k*-fold
cross-validation.
Boxes: `keyinsight`; `workedexample` a train/cv/test table for polynomial degrees 1–10; `pitfall`
selecting the degree on the test set; `researchspotlight` Stone (1974), Geman et al. (1992).
Reading: ISLR §5.1, §2.2; Bishop §1.3, §3.2.

### Ch 11 — Error Analysis, Data, Skew, and Ethics
The iterative loop of model development · error analysis by hand, and what it buys over a metric ·
adding data: augmentation, synthesis, data-centric practice · transfer learning and the two
fine-tuning regimes · skewed data, and why accuracy is useless at 0.5% prevalence · confusion
matrix, precision, recall, F1, and the trade-off curve · choosing a threshold from the cost of
each error type · fairness: sources of harm, subgroup evaluation, model cards.
**Scope boundary:** this chapter is the *modelling* loop. The *engineering* loop — pipelines,
serving, monitoring, retraining — is Part IX, and the two cross-reference each other.
Boxes: `workedexample` precision/recall/F1 from a stated confusion matrix; `ethicsnote` ×2;
`pitfall` optimising F1 when the two errors cost differently; `researchspotlight` Halevy, Norvig &
Pereira (2009), Buolamwini & Gebru (2018), Mitchell et al. (2019).
Reading: ISLR §4.4.2; Bishop §1.5.

### Ch 12 — Decision Trees and Ensembles  *(see gap G3)*
Trees as a sequence of questions · entropy, Gini, information gain, computed by hand · choosing a
split, including for continuous features · categorical encodings · stopping and pruning ·
regression trees · why one tree is high-variance · bagging · random forests and feature
subsampling · boosting and gradient-boosted trees · trees vs. neural networks on tabular data.
Boxes: `derivation` information gain; `workedexample` a full 10-row tree built by hand;
`algorithmbox` ×2; `researchspotlight` Quinlan (1986), Breiman (2001), Chen & Guestrin (2016).
Reading: ISLR §8.1–8.2; Bishop §14.4.

### Ch 13 — Clustering with *k*-Means
The algorithm: assign and move · the distortion objective, and the proof that neither step
increases it · convergence without global optimality · initialization and multiple restarts ·
*k*-means++ · choosing *K*: the elbow, its honest limits, and choosing from downstream cost ·
image compression as a worked application.
Boxes: `derivation` both steps as coordinate descent; `algorithmbox` ×2; `workedexample` two
iterations on six points by hand; `pitfall` the elbow that isn't; `researchspotlight` Lloyd (1982),
Arthur & Vassilvitskii (2007).
Reading: ISLR §12.4.1; Bishop §9.1.

### Ch 14 — Anomaly Detection
Density estimation with independent Gaussian features · MLE for μ and σ² · choosing the threshold
ε on a labelled validation set · evaluation under skew · anomaly detection vs. supervised
learning — the decision table · choosing and transforming features toward Gaussianity · the
multivariate Gaussian and how the two models relate.
Boxes: `derivation` MLE; `algorithmbox`; `workedexample` ε chosen on a small validation set with
F1 computed; `pitfall` "few positives ⇒ anomaly detection" applied without asking whether future
anomalies resemble past ones; `researchspotlight` Chandola, Banerjee & Kumar (2009).
Reading: ISLR §12.4 (context); Bishop §2.3.

### Ch 15 — Recommender Systems
Notation and the ratings matrix · content-based filtering with per-item features · collaborative
filtering: learning features and parameters together · its gradient · binary labels and the
logistic variant · mean normalization and the new-user problem · finding related items ·
content-based vs. collaborative, and cold start · two-tower deep retrieval · retrieval and ranking
for a large catalogue · what a recommender should not optimise.
Boxes: `derivation` the collaborative-filtering gradient; `workedexample` a 4×5 ratings matrix
factorised by hand for two iterations; `ethicsnote`; `pitfall` evaluating by RMSE alone;
`researchspotlight` Sarwar et al. (2001), Koren, Bell & Volinsky (2009), Covington et al. (2016).
Reading: ISLR §12.5 (context); Bishop §12.2.

### Ch 16 — Principal Component Analysis
The variance-maximising direction, derived with a Lagrange multiplier · eigenvectors and the SVD
route · the algorithm: centre, scale, decompose, project · explained variance and choosing the
number of components · reconstruction error · PCA for visualization, for compression, and misused
as a fix for overfitting · a note on t-SNE and UMAP for contrast.
Boxes: `derivation` the first component; `algorithmbox`; `workedexample` a 2D → 1D projection by
hand, checked against an SVD; `pitfall` PCA without centring or scaling; `researchspotlight`
Pearson (1901), van der Maaten & Hinton (2008).
Reading: ISLR §12.2; Bishop §12.1.

### Ch 17 — Reinforcement Learning and Deep Q-Networks
What makes RL different · states, actions, rewards, terminal states · the return and the discount
factor · policies · the MDP · `Q(s,a)` computed exhaustively on a small grid · the Bellman
equation, derived and interpreted · stochastic environments and expected return · continuous state
spaces · learning `Q` with a network: the training-set construction that turns RL into supervised
learning · DQN with a replay buffer · one output per action · ε-greedy exploration · mini-batch
and soft updates · where RL works, where it does not, and safety.
Boxes: `derivation` Bellman; `algorithmbox` DQN; `workedexample` the full `Q` table for a 6-state
world; `pitfall` a moving target without a target network; `ethicsnote` reward hacking;
`researchspotlight` Bellman (1957), Watkins & Dayan (1992), Mnih et al. (2015), Silver et al. (2016).
Reading: Sutton & Barto §3–6 (substitute; see the reading policy).

### Ch 18 — Sequence Models and the Transformer
Why sequences break the fixed-length feature vector · tokenization and subword vocabularies ·
word embeddings, and what "king − man + woman" really shows · recurrent networks in one section,
and their two failures: vanishing gradients and the sequential bottleneck · attention as a
learned, content-based lookup · queries, keys, values, and the scaled dot-product formula,
derived from that lookup analogy · why the `√d_k` is there · self-attention, worked by hand on a
four-token sentence · multi-head attention · positional encodings · the transformer block:
attention, feed-forward, residual connections, layer norm · encoder-only, decoder-only, and
encoder–decoder, and which tasks each suits · the causal mask · cost: quadratic in sequence
length, and what that means in practice.
Boxes: `derivation` scaled dot-product attention, reusing the softmax gradient from Ch 9;
`algorithmbox` a transformer block; `workedexample` self-attention over four tokens with numbers
computed in `verify/`; `pitfall` "attention weights are explanations"; `researchspotlight`
Mikolov et al. (2013), Bahdanau et al. (2015), Vaswani et al. (2017).
Prerequisites: Ch 9 (softmax, backprop) and Ch 16 (embeddings as low-dimensional representations).
Reading: Jurafsky & Martin 3e, ch. 9–11 (substitute).

### Ch 19 — Large Language Models in Practice
Next-token prediction as the pretraining objective, and why so much falls out of it · the training
pipeline: pretrain, supervised fine-tune, align · scaling laws in one figure · emergent behaviour
and the honest caveats · in-context learning, zero- and few-shot prompting · fine-tuning vs.
prompting vs. retrieval, as an engineering decision · retrieval-augmented generation, and why it
is a systems problem more than a modelling one · decoding: greedy, beam, temperature, top-*p* ·
evaluating generative systems, and why accuracy no longer applies · hallucination, and what the
mechanism actually is · cost, latency, and context limits · safety, misuse, provenance, and
licensing of training data.
Boxes: `workedexample` the same prompt decoded greedily and at three temperatures, with the
token probabilities computed in `verify/`; `algorithmbox` a retrieval-augmented answer pipeline;
`pitfall` treating a fluent answer as a sourced one; `ethicsnote` ×2; `keyinsight` "a language
model predicts text, and truth is not a term in its objective"; `researchspotlight`
Devlin et al. (2019), Brown et al. (2020), Lewis et al. (2020), Ouyang et al. (2022).
Prerequisite: Ch 18. This chapter is where Part IX's build-vs-buy question gets its modern answer,
so it is also the natural bridge into Ch 20.
Reading: Jurafsky & Martin 3e, ch. 12 and the LLM chapters (substitute).

---

## Part IX — Machine Learning Systems: Design, Development, Deployment

The pipeline, end to end. The distinction from Ch 10–11 is deliberate and stated in both places:
Ch 10–11 improve a *model*; Part IX builds and operates a *system* that contains one. Together
they answer the question every graduate is asked in their first job — "it works in your notebook,
now what?"

The running artefact for the whole part is one system, carried across all three chapters and
developed incrementally: a demand-forecasting service for a mid-sized retailer, chosen because it
needs batch training, low-latency serving, a feature store, and a retraining trigger, without
requiring any domain knowledge the reader lacks.

### Ch 20 — Designing a Machine Learning System
When *not* to use ML, as the first section, seriously argued · framing a business goal as a
learning problem: what is *x*, what is *y*, and who supplies the labels · from business metric to
model metric, and the gap between them · requirements: latency, throughput, freshness, cost,
interpretability, privacy — and the trade-offs among them · deciding batch vs. online prediction
before anything else, because it constrains everything downstream · the data strategy: sources,
volume, labelling cost, licensing · baselines you must beat: the constant, the heuristic, the
existing system · build vs. buy vs. call an API, and how the LLM era changed that calculus · the
reference architecture, drawn once and referred to for the rest of the part · writing a one-page
design document, with a filled-in example.
Boxes: `defbox` ×3; `keyinsight` "the metric you can measure is rarely the metric you care about";
`workedexample` the retailer's goal translated into a learning problem with a numeric target;
`pitfall` optimising a proxy metric to the point where it stops tracking the real one;
`ethicsnote` deciding who is affected before you build; `researchspotlight` Amershi et al. (2019),
Sculley et al. (2015).
Reading: Huyen, *Designing Machine Learning Systems*, ch. 1–2 (substitute).

### Ch 21 — The Development Pipeline: Data, Features, Training, Evaluation
Data ingestion and the sampling decisions that quietly bias everything · labelling: hand,
programmatic, weak supervision, and label noise · data validation as code — schema, range,
distribution — and failing the pipeline loudly · splits done right: random, temporal, and grouped,
and the leakage each prevents · feature engineering as pipeline code, not notebook code · the
training/serving skew problem and why a feature store exists · versioning data, features, code,
and models as four separate things · experiment tracking and reproducibility, including seeds and
nondeterminism · hyperparameter search: grid, random, and successive halving · the evaluation
suite: held-out metrics, slice-based metrics, behavioural tests, and a regression suite of known
failures · automating the pipeline, and what continuous integration means when the input is data ·
model packaging and the model registry.
Boxes: `algorithmbox` the pipeline as a DAG; `workedexample` a leakage bug that inflates
validation accuracy, with the corrected number; `pitfall` ×2 (leakage; tuning on the test set at
pipeline scale); `keyinsight` "a pipeline is reproducible or it is a story"; `labbox` build the
retailer's pipeline end to end; `researchspotlight` Breck et al. (2017), Polyzotis et al. (2018).
Reading: Huyen ch. 3–6 (substitute).

### Ch 22 — Deployment, Monitoring, and the Operating Model
Serving patterns: batch, request–response, streaming, and on-device, with the latency and cost
profile of each · a model behind an HTTP API, written out · scaling: batching, caching,
autoscaling, and the cost per thousand predictions · optimising inference: quantization, pruning,
distillation, and what each costs in accuracy · release strategy: shadow deployment, canary,
blue–green, and online A/B tests with a worked significance calculation · what to monitor —
operational health, input distribution, prediction distribution, and, when labels arrive,
accuracy · data drift and concept drift, distinguished, detected, and responded to · the
retraining decision: scheduled, triggered, or continual, and its cost · incident response and
rollback for a model · feedback loops, including the degenerate ones where the model's own outputs
become its training data · governance: model cards, audit trails, privacy, and access control ·
the team and the operating model — who owns the model at 3 a.m.
Boxes: `algorithmbox` a drift-detection check; `workedexample` an A/B test sized and read, numbers
computed in `verify/`; `pitfall` monitoring accuracy when labels arrive weeks late;
`keyinsight` "deployment is the beginning of the model's life, not the end of yours";
`ethicsnote` feedback loops that manufacture their own evidence; `labbox` serve, monitor, and
trigger a retrain; `researchspotlight` Paleyes et al. (2022), Sculley et al. (2015).
Reading: Huyen ch. 7–11 (substitute).

---

### Appendix A — Mathematical Background
Vectors, norms, dot products · matrices, multiplication, transpose, inverse, rank · linear systems
· derivatives and the chain rule · partial derivatives and gradients · the gradients used in the
book, tabulated · convexity · Lagrange multipliers (for Ch 16) · probability: random variables,
expectation, variance, the Gaussian, conditional probability, independence, maximum likelihood.
Cross-referenced *from* the chapter that first needs each item, so it is read in fragments.

### Appendix B — Python, NumPy, scikit-learn and TensorFlow
Environment setup · NumPy arrays, shapes, broadcasting, and the five shape bugs behind most lab
failures · vectorizing a loop step by step · matplotlib for the figures the book uses ·
scikit-learn's `fit`/`predict`/`transform` contract · Keras models, losses, optimizers · reading a
stack trace.

---

## Research Spotlights Map

| Paper | Chapter |
|---|---|
| Samuel (1959) — Checkers and the first "machine learning" | 1 |
| Mitchell (1997) — the T/P/E definition | 1 |
| Robbins & Monro (1951) — Stochastic approximation | 3 |
| Cox (1958) — The regression analysis of binary sequences | 6 |
| Rosenblatt (1958) — The Perceptron | 6, 8 |
| Hoerl & Kennard (1970) — Ridge regression | 7 |
| Tibshirani (1996) — Lasso | 7 |
| McCulloch & Pitts (1943) — A logical calculus of nervous activity | 8 |
| Cybenko (1989) — Universal approximation | 8 |
| Rumelhart, Hinton & Williams (1986) — Backpropagation | 9 |
| LeCun et al. (1998) — Gradient-based learning applied to document recognition | 9 |
| Nair & Hinton (2010) — ReLU | 9 |
| Kingma & Ba (2015) — Adam | 9 |
| Krizhevsky, Sutskever & Hinton (2012) — AlexNet | 9 |
| Stone (1974) — Cross-validation | 10 |
| Geman, Bienenstock & Doursat (1992) — The bias/variance dilemma | 10 |
| Domingos (2012) — A few useful things to know about ML | 10 |
| Halevy, Norvig & Pereira (2009) — The unreasonable effectiveness of data | 11 |
| Buolamwini & Gebru (2018) — Gender Shades | 11 |
| Mitchell et al. (2019) — Model Cards | 11, 22 |
| Quinlan (1986) — Induction of decision trees | 12 |
| Breiman (2001) — Random forests | 12 |
| Chen & Guestrin (2016) — XGBoost | 12 |
| Lloyd (1982) — Least squares quantization in PCM | 13 |
| Arthur & Vassilvitskii (2007) — k-means++ | 13 |
| Chandola, Banerjee & Kumar (2009) — Anomaly detection survey | 14 |
| Sarwar et al. (2001) — Item-based collaborative filtering | 15 |
| Koren, Bell & Volinsky (2009) — Matrix factorization for recommenders | 15 |
| Covington, Adams & Sargin (2016) — Deep neural networks for YouTube | 15 |
| Pearson (1901) — On lines and planes of closest fit | 16 |
| van der Maaten & Hinton (2008) — t-SNE | 16 |
| Bellman (1957) — Dynamic programming | 17 |
| Watkins & Dayan (1992) — Q-learning | 17 |
| Mnih et al. (2015) — Human-level control through deep RL | 17 |
| Silver et al. (2016) — AlphaGo | 17 |
| Mikolov et al. (2013) — word2vec | 18 |
| Bahdanau, Cho & Bengio (2015) — Neural MT by jointly learning to align and translate | 18 |
| Vaswani et al. (2017) — Attention is all you need | 18 |
| Devlin et al. (2019) — BERT | 19 |
| Brown et al. (2020) — Language models are few-shot learners | 19 |
| Lewis et al. (2020) — Retrieval-augmented generation | 19 |
| Ouyang et al. (2022) — Training language models to follow instructions | 19 |
| Amershi et al. (2019) — Software engineering for machine learning | 20 |
| Sculley et al. (2015) — Hidden technical debt in ML systems | 20, 22 |
| Breck et al. (2017) — The ML test score | 21 |
| Polyzotis et al. (2018) — Data management challenges in production ML | 21 |
| Paleyes, Urma & Lawrence (2022) — Challenges in deploying machine learning | 22 |

Forty-eight spotlights over twenty-two chapters, roughly two per chapter, one page each.

---

## Numerical integrity

*Any number stated as an expected answer must be computed, not guessed.* This binds the book more
tightly than the slides, because a textbook number gets quoted back in an exam.

- `verify/chNN_*.py` computes every number appearing in a `workedexample`, an iteration table, an
  error table, or an exercise answer, and prints it in the exact form used in the text.
- `verify/run_all.py` regenerates all of them; `make verify` runs it. A chapter is not done until
  its numbers match.
- Hand-worked iterations must agree with the script to the digits printed.
- Step sizes are dataset-specific and always stated with the dataset; the divergence threshold is
  derived, never asserted.
- Datasets are defined in Appendix D and shared with the labs, so a number in the book and the
  same number in a lab come from the same data.

---

## Course mapping (kept out of the book)

The book is course-neutral; this table is how the local 9-week delivery uses it. Update it here,
never in the chapter files.

| Module | Week | Session | Deck | Chapter | Lab notebooks |
|---|---|---|---|---|---|
| 1 | 1 | Part 1 | `cse815/W01_Part1` | Ch 1, Ch 2 | `cse816/W01_Part1–3` |
| 1 | 1 | Part 2 | `cse815/W01_Part2` | Ch 2 §4, Ch 3 | `cse816/W01_Part4` |
| 1 | 2 | Part 1 | `cse815/W02_Part1` | Ch 4 | `cse816/W02_Part1–2` |
| 1 | 2 | Part 2 | `cse815/W02_Part2` | Ch 5 | `cse816/W02_Part3–4` |
| 1 | 3 | Part 1 | `cse815/W03_Part1` | Ch 6 | M1W3 labs |
| 1 | 3 | Part 2 | `cse815/W03_Part2` | Ch 7 | M1W3 labs |
| 2 | 1 | Parts 1–2 | — | Ch 8 | M2W1 labs |
| 2 | 2 | Parts 1–2 | — | Ch 9 | M2W2 labs |
| 2 | 3 | Parts 1–2 | — | Ch 10, Ch 11 | M2W3 labs |
| 3 | 1 | Parts 1–2 | — | Ch 13, Ch 14 | M3W1 labs |
| 3 | 2 | Parts 1–2 | — | Ch 15, Ch 16 | M3W2 labs |
| 3 | 3 | Parts 1–2 | — | Ch 17 | M3W3 labs |
| — | — | reading / elective | — | Ch 12, Ch 18–19, Ch 20–22 | project |

Ch 18–19 and Part IX have no lecture slot in the current 9-week plan. They are what makes the book
worth reading beyond the course, and they are the natural home for a term project. If the syllabus
does expand to 12 weeks (gap G1), they are the three weeks to add.


### Notes from writing Ch 7–11 (2026-10-17)

- **New datasets** (defined in the chapters; belong in Appendix D): **T9**, ten training and nine
  test points from `0.5x² − 2x + 6` plus noise σ = 1.5, printed in full in Ch 7; **T10**, fifty
  points from `sin 2x + 0.5x` plus noise σ = 0.25, generated by a three-line seeded listing in
  Ch 10 and shared with the companion Week 6 lecture. Ch 8–9 reuse the lecture's 2-3-1 network
  (N231) so book and slides agree number for number. Ch 11's "screening model S" is analytic
  (two unit-variance normals, prevalence 0.5%), so its rates are exact.
- **Additions beyond the outline**: Ch 10 has a calibration section (reliability table, Brier
  score, log loss), because Ch 6 promised it; Ch 10 also measures the bias–variance terms by
  simulation. Ch 8 uses an XOR network (hand-set OR/NAND/AND) as its 2-2-1 hand-worked example.
- **`\va` added to `mlbook.sty`** (bold activation vector), matching the slide macro name.
- **G9** Ch 6, pitfall "Perfectly separable data has no maximum-likelihood fit", says the seven
  points without the 2 cm malignancy are "perfectly separated by x = 4.5". They are not: the 5 cm
  tumour is benign, so the separating gap is (5, 6), as the pitfall's own table (boundary → 5.49)
  shows. Ch 7 states the correct interval. Ch 6 not edited — raise with the author.
- **G10** Build environment: TeX Live is now 2026basic, but user-mode packages installed for
  2025 in `~/Library/texmf` (notably `biblatex` 3.20 and `etoolbox` 2.5k) shadow the system
  versions and fail with "Argument of \reserved@a has an extra }". Workaround used while writing:
  `TEXINPUTS=/usr/local/texlive/2026basic/texmf-dist/tex/latex/biblatex//:/usr/local/texlive/2026basic/texmf-dist/tex/latex/etoolbox//: make`.
  Permanent fix: remove or rename those two directories under `~/Library/texmf/tex/latex/`.

### Notes from writing Ch 12–14 (2026-10-19)

- **G3 resolved by request**: the author asked for Ch 12; it is written as a full chapter and
  can be assigned as reading even though no lecture week covers trees.
- **New datasets**: **T11** (ten loan applicants, three binary features plus income; Ch 12),
  **K6** (six points, Ch 13 hand example), **B60** and **U60** (Ch 13; B60 is the companion
  Week 7 lecture's data), **E2** (engines) and **DC** (data-centre machines) in Ch 14, both
  shared with the Week 7 lecture. Ch 13's image example uses scikit-learn's bundled
  `china.jpg`; the 16-colour palette is drawn as TikZ swatches (no raster image).
- Ch 12's regression trees on T10 keep improving with depth on its 15-point test set; the text
  says so rather than claiming overfitting, and compares with Ch 10's cubic instead.
- Ch 14 reads ISLR §4.4.1–4.4.2 (Gaussian densities) rather than the outline's "§12.4
  (context)", which does not cover the material.

### Known gaps to resolve (raise, do not silently fix)

- **G1** `plan.md` headers say 12 weeks; 9 weeks of content are listed. Ch 12, Ch 18–19 and
  Part IX are exactly the material that would fill the gap.
- **G2** `index.html` still carries the older 13-week syllabus, and its Research Articles section
  is tagged against that numbering.
- **G3** The Module 2 title promises decision trees; no week lists them and no deck exists. Ch 12
  is written as if they belong. Decision needed: add a week, or make Ch 12 optional reading.
- **G4** PCA is optional in `plan.md` but is a full chapter here (Ch 16). It is short and
  self-contained, so it can be assigned as reading.
- **G5** Four chapters name a substitute third book (Sutton & Barto, Jurafsky & Martin, Huyen)
  rather than adding to the two-book budget. Confirm this reading of the standing instruction.
- **G6** ~~Ch 18 at 52 pages~~ **RESOLVED 2026-09-03:** split into Ch 18 (sequences, attention,
  the transformer) and Ch 19 (large language models in practice). Everything after shifts by one.
- **G7** ~~Running system for Part IX~~ **RESOLVED 2026-09-03:** retail demand forecasting,
  confirmed.
- **G8** ~~`W01_Part1.tex` draws the flat-price fit as `3.6x + 3`~~ **RESOLVED 2026-09-03:** the
  deck now draws the least-squares fit `3.2984x + 4.3308` and annotates 60.4 at `x = 17`, with a
  comment pointing at `book/verify/ch01_flat_prices.py`. Rebuilt; `make check` clean.

---

## Writing Order and Milestones

| M | Deliverable | Contents |
|---|---|---|
| **M0** | Scaffolding | `mlbook.sty`, `main.tex`, `preface.tex`, `notation.tex`, `Makefile`, `bibliography.bib`, `verify/`, chapter stubs. Builds. **DONE** |
| **M1** | Part I (Ch 1–5) | **COMPLETE.** Pattern-setting part; the chapter template is frozen here. |
| **M2** | Part II (Ch 6–7) | **COMPLETE.** Completes the linear-model core. |
| **M3** | Appendices A–B | Written early; Ch 8 onward leans on them. |
| **M4** | Part III (Ch 8–9) | **COMPLETE.** Heaviest derivations. |
| **M5** | Part IV (Ch 10–12) | **COMPLETE.** Ch 12 written on request (see G3). |
| **M6** | Parts V–VII (Ch 13–17) | Ch 13–14 **DONE**. |
| **M7** | Ch 18–19 | Depend on Ch 9 and Ch 16 being final; Ch 19 depends on Ch 18. |
| **M8** | Part IX (Ch 20–22) | Depends on Ch 10–11 for the modelling/engineering boundary. |
| **M9** | Back matter and polish | Appendices C–E, index pass, `make check` clean, cover. |

Per-chapter definition of done:

1. All sections written; every `derivation` complete, with no "it can be shown that".
2. Every figure a TikZ/pgfplots source, legible in greyscale, `forget plot` on unlabelled scatters.
3. `chapterlearning` objectives each traceable to a section and exercised by an exercise.
4. SQ/AP/DP/PE/QZ quotas met, including at least one open-ended design question.
5. Numbers regenerated from `verify/` and matching.
6. Further Reading names ISLR and, where used, Bishop, with section numbers — or the one declared
   substitute.
7. No course code, institution, week number, or local currency in the prose.
8. `make check` clean: no overfull box over 5 pt, no undefined reference or citation.
