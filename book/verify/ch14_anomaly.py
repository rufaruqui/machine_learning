"""Chapter 14 --- numbers for anomaly detection.

Datasets (shared with the companion lecture):
  E2   aircraft engines: heat (deg C) and vibration (mm/s).  6000 good engines
       for training; validation and test sets of 2000 good + 10 faulty each.
       Good engines ~ N((500, 4), diag(25^2, 0.5^2)); faulty ones lie 3.2-5
       standard deviations from the centre in a random direction.
  L1   1000 readings of a right-skewed feature, exp(N(0, 0.8^2)).
  DC   500 data-centre machines: CPU load and network traffic, strongly
       correlated, plus one stuck machine at (85, 250).

Run:  /tmp/mlenv/bin/python verify/ch14_anomaly.py
"""

from math import log, pi, sqrt

import numpy as np

np.set_printoptions(precision=4, suppress=True)

r = np.random.default_rng(7)
mu_t = np.array([500.0, 4.0]); sd_t = np.array([25.0, 0.5])
Xtr = mu_t + sd_t * r.normal(size=(6000, 2))


def bad(n, seed):
    q = np.random.default_rng(seed)
    ang = q.uniform(0, 2 * np.pi, n); rad = q.uniform(3.2, 5.0, n)
    return mu_t + sd_t * np.c_[rad * np.cos(ang), rad * np.sin(ang)]


Xcv = np.r_[mu_t + sd_t * r.normal(size=(2000, 2)), bad(10, 1)]; ycv = np.r_[np.zeros(2000), np.ones(10)]
Xte = np.r_[mu_t + sd_t * r.normal(size=(2000, 2)), bad(10, 2)]; yte = np.r_[np.zeros(2000), np.ones(10)]

mu = Xtr.mean(0); var = Xtr.var(0); sd = np.sqrt(var)
print("[14.3] maximum-likelihood estimates from 6000 engines")
print(f"  mu = {mu}, sigma^2 = {var}, sigma = {sd}")
print(f"  with 1/(m-1): sigma^2 = {Xtr.var(0, ddof=1)}")


def p_each(X):
    return np.exp(-(X - mu) ** 2 / (2 * var)) / np.sqrt(2 * pi * var)


def p(X):
    return np.prod(p_each(X), axis=1)


print("\n[14.4] three new engines")
for x in ([505, 4.1], [560, 4.9], [590, 5.6]):
    x = np.array([x], float)
    print(f"  {x[0]}: z = {((x - mu) / sd).round(2)[0]}, p(x_j) = {p_each(x).round(6)[0]}, p(x) = {p(x)[0]:.3e}")


def prf(eps, X, y):
    yh = p(X) < eps
    tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int(((~yh) & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else float("nan"); R = tp / (tp + fn)
    F = 2 * P * R / (P + R) if tp else 0.0
    return tp, fp, fn, P, R, F


print("\n[14.5] epsilon sweep on the validation set")
for eps in [1e-3, 1e-4, 3e-5, 1e-5, 3e-6]:
    tp, fp, fn, P, R, F = prf(eps, Xcv, ycv)
    q = -2 * log(eps * 2 * pi * sd[0] * sd[1])
    print(f"  eps = {eps:.0e}: TP {tp:2d} FP {fp:3d} FN {fn:2d}  P {P:.3f} R {R:.1f} F1 {F:.3f}"
          f"   boundary radius {sqrt(q):.3f} sd")
tp, fp, fn, P, R, F = prf(1e-5, Xte, yte)
print(f"  test at eps = 1e-5: TP {tp} FP {fp} FN {fn}  P {P:.3f} R {R:.3f} F1 {F:.3f}")
print(f"  fraction of training engines flagged at 1e-5: {(p(Xtr) < 1e-5).mean():.5f}")

print("\n[14.5] choosing epsilon by hand on a 12-engine validation set (9 good, 3 faulty)")
rs = np.random.default_rng(14)
good = rs.choice(2000, 9, replace=False); badi = 2000 + rs.choice(10, 3, replace=False)
idx = np.r_[good, badi]
ps = p(Xcv[idx]); ys = ycv[idx]
order = np.argsort(ps)
for k in order:
    print(f"  engine {Xcv[idx][k].round(1)}  y = {int(ys[k])}  p = {ps[k]:.3e}")
cands = (np.sort(ps)[:-1] * np.sort(ps)[1:]) ** 0.5     # geometric midpoints
best = None
for e in cands:
    yh = ps < e
    tp = int((yh & (ys == 1)).sum()); fp = int((yh & (ys == 0)).sum()); fn = int(((~yh) & (ys == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn); F = 2 * P * R / (P + R) if tp else 0
    print(f"  eps = {e:.3e}: TP {tp} FP {fp} FN {fn}  P {P:.3f} R {R:.3f} F1 {F:.3f}")
    if best is None or F > best[1]:
        best = (e, F)
print(f"  best: eps = {best[0]:.3e}, F1 = {best[1]:.3f}")

print("\n[14.7] a skewed feature and its logarithm")
q = np.random.default_rng(3); s = np.exp(q.normal(0, 0.8, 1000))
h, e = np.histogram(s, bins=12, range=(0, 6)); h2, _ = np.histogram(np.log(s), bins=12, range=(-3, 3))
print("  raw counts:", h.tolist(), " values above 6:", int((s > 6).sum()))
print("  log counts:", h2.tolist())
from scipy import stats as st  # skewness only
print(f"  skewness raw {st.skew(s):.3f}, log {st.skew(np.log(s)):.3f}")

print("\n[14.7-14.8] correlated features: the stuck machine")
q = np.random.default_rng(5); load = q.uniform(20, 90, 500); traffic = load * q.normal(1.0, 0.08, 500) * 10
Xd = np.c_[load, traffic]; an = np.array([[85.0, 250.0]])
m2, v2 = Xd.mean(0), Xd.var(0)
pp = lambda X, m, v: np.prod(np.exp(-(X - m) ** 2 / (2 * v)) / np.sqrt(2 * pi * v), axis=1)
pa = pp(an, m2, v2)[0]
print(f"  mean {m2}, sd {np.sqrt(v2)}; anomaly z = {((an - m2) / np.sqrt(v2)).round(2)[0]}")
print(f"  independent model: p(anomaly) = {pa:.3e}; normal machines with lower p: {(pp(Xd, m2, v2) < pa).sum()} of 500")
rat = Xd[:, 0] / Xd[:, 1]; ra = 85 / 250
print(f"  ratio feature: normal {rat.mean():.4f} +- {rat.std():.4f}; anomaly {ra:.2f}, z = {(ra - rat.mean()) / rat.std():.1f}")
S = np.cov(Xd.T, bias=True)
Si = np.linalg.inv(S)
def pmv(X):
    d = X - m2
    q = np.einsum("ij,jk,ik->i", d, Si, d)
    return np.exp(-q / 2) / (2 * pi * np.sqrt(np.linalg.det(S))), q
pm_an, q_an = pmv(an); pm_n, q_n = pmv(Xd)
corr = S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])
print(f"  covariance matrix {S.round(1).tolist()}, correlation {corr:.4f}")
print(f"  multivariate: p(anomaly) = {pm_an[0]:.3e} (Mahalanobis^2 {q_an[0]:.1f});"
      f" least likely normal machine p = {pm_n.min():.3e} (Mahalanobis^2 {q_n.max():.2f})")
print(f"  parameters: independent 2n = {2 * 2}, multivariate n + n(n+1)/2 = {2 + 3};"
      f" for n = 100: {200} vs {100 + 100 * 101 // 2}")
