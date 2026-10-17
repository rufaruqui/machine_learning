"""Chapter 12 --- numbers for decision trees and ensembles.

Datasets:
  T11  ten loan applicants: employed, owns home, good credit history (binary),
       income in thousands of dollars, and whether the loan was repaid.
  T10  the synthetic regression set of Chapter 10 (regression trees).
  M2   two interleaved half-moons, 400 training / 2000 test points, noise 0.3
       (sklearn make_moons, random_state 12 and 13): ensembles.
  G6   six points for gradient boosting by hand.

Run:  /tmp/mlenv/bin/python verify/ch12_trees.py
"""

from math import log2

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def H(p):
    return 0.0 if p in (0.0, 1.0) else -p * log2(p) - (1 - p) * log2(1 - p)


def gini(p):
    return 2 * p * (1 - p)


T11 = np.array([
    [1, 1, 1, 62, 1], [1, 0, 1, 48, 1], [0, 0, 1, 35, 0], [1, 1, 0, 55, 1],
    [0, 1, 1, 41, 1], [1, 0, 0, 30, 0], [0, 0, 0, 22, 0], [1, 0, 1, 52, 1],
    [0, 1, 0, 38, 0], [1, 0, 0, 45, 0]], float)
NAMES = ["employed", "owns home", "good history"]


def split_gain(mask, y, imp=H):
    n = len(y)
    l, r = y[mask], y[~mask]
    w = lambda s: (len(s) / n) * imp(s.mean()) if len(s) else 0.0
    return imp(y.mean()) - w(l) - w(r), (len(l), int(l.sum())), (len(r), int(r.sum()))


print("[12.2] impurity values")
for p in (0, 0.2, 0.5, 0.8, 1.0, 4 / 5, 1 / 5):
    print(f"  p = {p:.3f}: entropy {H(p):.4f}, gini {gini(p):.4f}")

y = T11[:, 4]
print(f"\n[12.3] T11 root: {int(y.sum())} of 10 repaid, H = {H(y.mean()):.4f}")
for j, nm in enumerate(NAMES):
    g, L, R = split_gain(T11[:, j] == 1, y)
    gg, _, _ = split_gain(T11[:, j] == 1, y, gini)
    print(f"  {nm:12}: yes {L}, no {R}; information gain {g:.4f}; gini reduction {gg:.4f}")


def build(rows, feats, depth=0, path="root"):
    yy = T11[rows, 4]
    pad = "  " * (depth + 1)
    if yy.mean() in (0, 1) or not feats:
        print(f"{pad}{path}: leaf, {len(rows)} examples, {int(yy.sum())} repaid -> predict {int(round(yy.mean()))}")
        return
    best = max(feats, key=lambda j: split_gain(T11[rows, j] == 1, yy)[0])
    g, L, R = split_gain(T11[rows, best] == 1, yy)
    print(f"{pad}{path}: {len(rows)} examples ({int(yy.sum())} repaid), H = {H(yy.mean()):.4f};"
          f" split on '{NAMES[best]}' (gain {g:.4f})")
    for j in feats:
        if j != best:
            gj, Lj, Rj = split_gain(T11[rows, j] == 1, yy)
            print(f"{pad}   alternative '{NAMES[j]}': gain {gj:.4f}  yes {Lj} no {Rj}")
    rest = [j for j in feats if j != best]
    yes = [r for r in rows if T11[r, best] == 1]
    no = [r for r in rows if T11[r, best] == 0]
    build(yes, rest, depth + 1, f"{NAMES[best]} = yes")
    build(no, rest, depth + 1, f"{NAMES[best]} = no")


print("\n[12.3] the full tree (binary features only)")
build(list(range(10)), [0, 1, 2])

print("\n[12.4] continuous feature: income thresholds at the root")
xs = np.sort(np.unique(T11[:, 3]))
for t in (xs[:-1] + xs[1:]) / 2:
    g, L, R = split_gain(T11[:, 3] > t, y)
    print(f"  income > {t:5.1f}: gain {g:.4f}  above {L} below {R}")

print("\n[12.6] regression trees on T10")
rng = np.random.default_rng(4)
x = rng.uniform(0, 3, 50)
yr = np.sin(2 * x) + 0.5 * x + rng.normal(0, 0.25, 50)
xtr, ytr, xte, yte = x[:35], yr[:35], x[35:], yr[35:]
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
J = lambda p, t: float(np.mean((p - t) ** 2) / 2)
for d in (1, 2, 3, 5, None):
    m = DecisionTreeRegressor(max_depth=d, random_state=0).fit(xtr[:, None], ytr)
    print(f"  max_depth {str(d):4}: leaves {m.get_n_leaves():2d}  J_train {J(m.predict(xtr[:, None]), ytr):.4f}"
          f"  J_test {J(m.predict(xte[:, None]), yte):.4f}")
m1 = DecisionTreeRegressor(max_depth=1).fit(xtr[:, None], ytr)
t1 = m1.tree_
print(f"  depth-1 split: x <= {t1.threshold[0]:.4f}; left mean {t1.value[1][0][0]:.4f} (n={t1.n_node_samples[1]}),"
      f" right mean {t1.value[2][0][0]:.4f} (n={t1.n_node_samples[2]})")
print(f"  variance at root {np.var(ytr):.4f}")
m2 = DecisionTreeRegressor(max_depth=2).fit(xtr[:, None], ytr)
grid = np.linspace(0, 3, 301)
for nm, m in [("d2", m2), ("dfull", DecisionTreeRegressor(random_state=0).fit(xtr[:, None], ytr))]:
    p = m.predict(grid[:, None])
    print(f"  curve {nm}:", "".join(f"({a:.3f},{b:.3f})" for a, b in zip(grid[::3], p[::3])))

print("\n[12.6] variance reduction by hand: six points")
xs6 = np.array([1, 2, 3, 4, 5, 6.0]); ys6 = np.array([1.0, 1.4, 1.2, 3.8, 4.2, 4.0])
for t in (1.5, 2.5, 3.5, 4.5, 5.5):
    L, R = ys6[xs6 < t], ys6[xs6 > t]
    red = np.var(ys6) - len(L) / 6 * np.var(L) - len(R) / 6 * np.var(R)
    print(f"  x < {t}: left mean {L.mean():.4f}, right mean {R.mean():.4f}, variance reduction {red:.4f}")
print(f"  root variance {np.var(ys6):.4f}")

print("\n[12.7] one tree is unstable: root split under bootstrap resampling of T11")
from collections import Counter
cnt = Counter()
for s in range(1000):
    idx = np.random.default_rng(s).integers(0, 10, 10)
    yy = T11[idx, 4]
    if yy.mean() in (0, 1):
        cnt["pure"] += 1
        continue
    best = max(range(4), key=lambda j: max(
        (split_gain(T11[idx, j] > t, yy)[0] for t in ([0.5] if j < 3 else (np.sort(np.unique(T11[idx, 3]))[:-1] + 0.5))),
        default=0))
    cnt[(NAMES + ["income"])[best]] += 1
print("  root feature over 1000 bootstrap samples:", dict(cnt))

print("\n[12.8-12.10] ensembles on two moons (400 train, 2000 test)")
from sklearn.datasets import make_moons
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
Xm, ym = make_moons(400, noise=0.3, random_state=12)
Xt, yt = make_moons(2000, noise=0.3, random_state=13)
models = {
    "one full tree": DecisionTreeClassifier(random_state=0),
    "one tree, depth 4": DecisionTreeClassifier(max_depth=4, random_state=0),
    "bagging, 200 trees": BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=0),
    "random forest, 200 trees": RandomForestClassifier(n_estimators=200, max_features=1, random_state=0),
    "gradient boosting, 200 stumps": GradientBoostingClassifier(n_estimators=200, max_depth=1, learning_rate=0.3, random_state=0),
    "gradient boosting, 200 depth-3": GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=0),
}
for nm, m in models.items():
    m.fit(Xm, ym)
    print(f"  {nm:32}: train acc {m.score(Xm, ym):.4f}  test acc {m.score(Xt, yt):.4f}")
# spread of single full trees on bootstrap samples
accs = []
for s in range(50):
    idx = np.random.default_rng(100 + s).integers(0, 400, 400)
    accs.append(DecisionTreeClassifier(random_state=0).fit(Xm[idx], ym[idx]).score(Xt, yt))
print(f"  single trees on 50 bootstrap samples: test acc mean {np.mean(accs):.4f}, min {np.min(accs):.4f}, max {np.max(accs):.4f}")
from sklearn.linear_model import LogisticRegression
print(f"  logistic regression baseline: test acc {LogisticRegression().fit(Xm, ym).score(Xt, yt):.4f}")
print("  bagging test accuracy vs number of trees:")
for B in (1, 5, 10, 25, 50, 100, 200):
    acc = BaggingClassifier(DecisionTreeClassifier(), n_estimators=B, random_state=0).fit(Xm, ym).score(Xt, yt)
    print(f"    B = {B:3d}: {acc:.4f}")

print("\n[12.10] gradient boosting by hand (squared error, stumps, learning rate 0.5)")
xg = np.array([1, 2, 3, 4, 5, 6.0]); yg = np.array([1.0, 1.2, 2.6, 3.0, 4.4, 4.8])
F = np.full(6, yg.mean())
print(f"  F0 = mean = {yg.mean():.4f}; residuals {yg - F}")
for rnd in (1, 2, 3):
    r = yg - F
    best = None
    for t in (1.5, 2.5, 3.5, 4.5, 5.5):
        L, R = r[xg < t], r[xg > t]
        sse = ((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, t, L.mean(), R.mean())
    _, t, lv, rv = best
    h = np.where(xg < t, lv, rv)
    F = F + 0.5 * h
    print(f"  round {rnd}: stump x < {t}: left {lv:.4f}, right {rv:.4f}; F = {np.round(F, 4)};"
          f" SSE = {((yg - F) ** 2).sum():.4f}")
