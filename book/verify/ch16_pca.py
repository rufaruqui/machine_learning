"""Chapter 16 --- numbers for principal component analysis.

Datasets:
  P5   five centred points in the plane, worked by hand
  C30  30 points, two correlated features, standardised (shared with the
       companion lecture's axis-search animation)
  DIG  sklearn's 8x8 handwritten digits (1797 images, 64 features)

Run:  /tmp/mlenv/bin/python verify/ch16_pca.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

print("[16.4] five points by hand")
P5 = np.array([[2, 1], [1, 1], [0, 0], [-1, -1], [-2, -1]], float)
print(f"  mean {P5.mean(0)}")
S = P5.T @ P5 / 5
print("  covariance (1/m) X^T X =\n", S)
tr, det = np.trace(S), np.linalg.det(S)
disc = np.sqrt(tr ** 2 - 4 * det)
l1, l2 = (tr + disc) / 2, (tr - disc) / 2
print(f"  trace {tr:.4f}, det {det:.4f}, sqrt(trace^2 - 4 det) {disc:.4f}; eigenvalues {l1:.4f}, {l2:.4f}")
u = np.array([1.2, l1 - 2.0]); u /= np.linalg.norm(u)
print(f"  u1 from (Sigma - l1 I)u = 0: u2/u1 = {(l1 - 2) / 1.2:.4f}; u1 = {u}")
print(f"  golden ratio check: (sqrt(5)-1)/2 = {(np.sqrt(5) - 1) / 2:.4f}")
z = P5 @ u
print(f"  projections z = {z.round(4)}; variance of z = {np.mean(z ** 2):.4f}")
print(f"  reconstructions:\n{np.outer(z, u).round(4)}")
print(f"  variance explained {l1 / (l1 + l2):.4f}; mean squared reconstruction error {np.mean(((P5 - np.outer(z, u)) ** 2).sum(1)):.4f}")
U_, s_, Vt = np.linalg.svd(P5, full_matrices=False)
print(f"  SVD: singular values {s_.round(4)}, s^2/m = {(s_ ** 2 / 5).round(4)}, first right singular vector {Vt[0].round(4)}")
print(f"  least-squares regression of x2 on x1: slope {np.sum(P5[:, 0] * P5[:, 1]) / np.sum(P5[:, 0] ** 2):.4f};"
      f" PCA direction slope {u[1] / u[0]:.4f}")

print("\n[16.2] C30: variance of projections against axis angle")
rng = np.random.default_rng(8)
t = rng.normal(0, 1, 30)
C = np.c_[t + rng.normal(0, 0.35, 30), 0.8 * t + rng.normal(0, 0.35, 30)]
C = (C - C.mean(0)) / C.std(0)
for a in (0, 15, 30, 45, 60, 90, 135):
    v = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
    print(f"  angle {a:3d}: variance {np.mean((C @ v) ** 2):.4f}")
ev = np.linalg.eigvalsh(C.T @ C / 30)[::-1]
print(f"  eigenvalues {ev.round(4)}; correlation {np.mean(C[:, 0] * C[:, 1]):.4f}")
print("  curve:", "".join(f"({a},{np.mean((C @ [np.cos(np.radians(a)), np.sin(np.radians(a))]) ** 2):.4f})"
                          for a in range(0, 181, 5)))
print("  points:", "".join(f"({a:.3f},{b:.3f})" for a, b in C))

print("\n[16.6] digits")
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
dg = load_digits()
X, y = dg.data / 16.0, dg.target
pca = PCA().fit(X)
cum = np.cumsum(pca.explained_variance_ratio_)
for k in (1, 2, 5, 10, 15, 20, 30, 40, 64):
    print(f"  k = {k:2d}: variance kept {cum[k - 1]:.4f}")
print(f"  k for 90%: {np.argmax(cum >= 0.9) + 1}, for 95%: {np.argmax(cum >= 0.95) + 1}, for 99%: {np.argmax(cum >= 0.99) + 1}")
print("  cumulative curve:", "".join(f"({k},{cum[k - 1]:.4f})" for k in range(1, 41)))
sel = np.isin(y, [0, 1, 7])
p3 = PCA(2).fit(X[sel]); Z3 = p3.transform(X[sel])
print(f"  digits 0, 1, 7 only ({sel.sum()} images): two components keep {p3.explained_variance_ratio_.sum():.4f}")
rs = np.random.default_rng(0)
yy = y[sel]
keep = np.concatenate([rs.choice(np.flatnonzero(yy == c), 40, replace=False) for c in (0, 1, 7)])
for c in (0, 1, 7):
    pts = "".join(f"({Z3[i, 0]:.3f},{Z3[i, 1]:.3f})" for i in keep if yy[i] == c)
    print(f"  scatter{c}: {pts}")

print("\n[16.5] reconstructing one image of a 3")
i3 = np.flatnonzero(y == 3)[0]
x3 = X[i3]
print("  original:", np.round(x3 * 16).astype(int).tolist())
for k in (2, 5, 15, 64):
    p = PCA(k).fit(X)
    xh = np.clip(p.inverse_transform(p.transform(x3[None]))[0], 0, 1)
    err = float(np.mean((xh - x3) ** 2))
    print(f"  k = {k:2d}: squared error per pixel {err:.4f}; image {np.round(xh * 16).astype(int).tolist()}")
for k in (2, 5, 10, 15, 30):
    p = PCA(k).fit(X)
    Xh = p.inverse_transform(p.transform(X))
    print(f"  k = {k:2d}: mean squared reconstruction error over all images {np.mean((Xh - X) ** 2):.5f}"
          f"; storage {k}/64 = {k / 64:.3f} of the original")

print("\n[16.7] PCA is not a cure for overfitting (digits, 10 classes)")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=16, stratify=y)
Xtr, ytr = Xtr[:200], ytr[:200]          # a small training set, where overfitting threatens
print(f"  training on {len(ytr)} images, testing on {len(yte)}")
for C_ in (1e4, 1.0, 0.1):
    m = LogisticRegression(C=C_, max_iter=20000).fit(Xtr, ytr)
    print(f"  all 64 pixels, C = {C_:g} (lambda = 1/C): train {m.score(Xtr, ytr):.4f}, test {m.score(Xte, yte):.4f}")
for k in (5, 10, 20):
    p = PCA(k).fit(Xtr)
    m = LogisticRegression(C=1e4, max_iter=20000).fit(p.transform(Xtr), ytr)
    print(f"  PCA to {k:2d}, unregularised: train {m.score(p.transform(Xtr), ytr):.4f},"
          f" test {m.score(p.transform(Xte), yte):.4f}")
