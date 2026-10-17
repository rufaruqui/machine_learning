"""Chapter 7 --- numbers for overfitting and regularization.

Datasets:
  T9   ten training and nine test points drawn from y = 0.5x^2 - 2x + 6
       plus Gaussian noise (sigma = 1.5), rounded to one decimal.  The
       values are printed in the chapter, so the seed is only how they were
       first produced.
  T7s  T7 of Chapter 6 with the 2 cm malignancy removed (linearly separable)
  T8   twenty tumours, size and texture (Chapter 6)

Conventions (match the text):
  linear cost   J = (1/2m) sum (f - y)^2 + (lam/2m) sum_j w_j^2
  L1 cost       J = (1/2m) sum (f - y)^2 + (lam/m)  sum_j |w_j|
  logistic      J = mean logistic loss + (lam/2m) sum_j w_j^2
  b is never penalised; polynomial features are standardised on the
  training set before fitting.

Run:  /tmp/mlenv/bin/python verify/ch07_regularization.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)

# --------------------------------------------------------------- datasets
_f = lambda x: 0.5 * x ** 2 - 2 * x + 6
_r = np.random.default_rng(2)
T9x_tr = np.arange(1.0, 11.0)
T9y_tr = np.round(_f(T9x_tr) + _r.normal(0, 1.5, 10), 1)
T9x_te = np.arange(1.5, 10.0, 1.0)
T9y_te = np.round(_f(T9x_te) + _r.normal(0, 1.5, 9), 1)

T7x = np.array([1.0, 3, 4, 5, 6, 7, 8])
T7y = np.array([0.0, 0, 0, 0, 1, 1, 1])

T8 = np.array([
    [2.4, 3.4, 0], [1.8, 0.4, 0], [2.2, 4.6, 0], [2.9, 2.7, 0],
    [3.8, 5.9, 0], [2.0, 3.2, 0], [2.9, 3.1, 0], [2.4, 5.0, 0],
    [3.0, 2.7, 0], [1.9, 2.7, 0], [3.8, 6.3, 1], [3.5, 7.2, 1],
    [3.8, 6.2, 1], [2.4, 4.4, 1], [3.3, 6.7, 1], [3.2, 8.2, 1],
    [3.9, 6.5, 1], [3.9, 6.8, 1], [4.4, 5.5, 1], [4.9, 5.2, 1],
])


# --------------------------------------------------------------- machinery
class Poly:
    """Powers x, x^2, ..., x^d, standardised with training statistics."""

    def __init__(self, x, d):
        self.d = d
        P = self.raw(x)
        self.mu, self.sd = P.mean(0), P.std(0)

    def raw(self, x):
        return np.vstack([x ** k for k in range(1, self.d + 1)]).T

    def __call__(self, x):
        return (self.raw(x) - self.mu) / self.sd


def ridge(Z, y, lam):
    """Minimise (1/2m)||b + Zw - y||^2 + (lam/2m)||w||^2; b unpenalised."""
    m, n = Z.shape
    A = np.c_[np.ones(m), Z]
    R = lam * np.eye(n + 1)
    R[0, 0] = 0.0
    th = np.linalg.solve(A.T @ A + R, A.T @ y)
    return th[0], th[1:]


def J_sq(Z, y, b, w):
    return float(np.mean((b + Z @ w - y) ** 2) / 2)


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def logistic_fit(X, y, lam, iters=200):
    """Newton's method on mean logistic loss + (lam/2m)||w||^2."""
    m, n = X.shape
    A = np.c_[np.ones(m), X]
    th = np.zeros(n + 1)
    R = lam / m * np.eye(n + 1)
    R[0, 0] = 0.0
    for _ in range(iters):
        p = sig(A @ th)
        g = A.T @ (p - y) / m + R @ th
        H = A.T @ (A * (p * (1 - p))[:, None]) / m + R
        th -= np.linalg.solve(H + 1e-12 * np.eye(n + 1), g)
    return th


def logistic_J(X, y, th):
    A = np.c_[np.ones(len(y)), X]
    p = np.clip(sig(A @ th), 1e-300, 1 - 1e-16)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


# ======================================================================
print("[T9] the data")
print("  train x:", T9x_tr)
print("  train y:", T9y_tr)
print("  test  x:", T9x_te)
print("  test  y:", T9y_te)

print("\n[7.1] degree sweep, unregularised")
fits = {}
for d in range(1, 10):
    P = Poly(T9x_tr, d)
    b, w = ridge(P(T9x_tr), T9y_tr, 0.0)
    fits[d] = (P, b, w)
    print(f"  degree {d}:  J_train = {J_sq(P(T9x_tr), T9y_tr, b, w):9.4f}"
          f"   J_test = {J_sq(P(T9x_te), T9y_te, b, w):9.4f}"
          f"   max|w| = {np.abs(w).max():10.3f}")

print("\n[7.1] figure curves (x, f(x)) for degrees 1, 2, 9")
grid = np.round(np.arange(0.8, 10.21, 0.2), 2)
for d in (1, 2, 9):
    P, b, w = fits[d]
    v = b + P(grid) @ w
    print(f"  D{d}:", "".join(f"({a:.1f},{c:.3f})" for a, c in zip(grid, v)))

print("\n[7.4] lambda sweep, degree 9 (standardised powers)")
P9 = fits[9][0]
Ztr, Zte = P9(T9x_tr), P9(T9x_te)
for lam in [0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1000]:
    b, w = ridge(Ztr, T9y_tr, lam)
    print(f"  lam = {lam:>7g}:  J_train = {J_sq(Ztr, T9y_tr, b, w):8.4f}"
          f"   J_test = {J_sq(Zte, T9y_te, b, w):8.4f}"
          f"   ||w|| = {np.linalg.norm(w):10.3f}   b = {b:8.4f}")
best = min([10 ** k for k in np.arange(-4, 3.01, 0.25)],
           key=lambda l: J_sq(Zte, T9y_te, *ridge(Ztr, T9y_tr, l)))
print(f"  finer grid: test cost minimised near lam = {best:.4g},"
      f" J_test = {J_sq(Zte, T9y_te, *ridge(Ztr, T9y_tr, best)):.4f}")

print("\n[7.4] lambda curves for the figure")
for lam in [10 ** k for k in np.arange(-4, 3.01, 0.5)]:
    b, w = ridge(Ztr, T9y_tr, lam)
    print(f"  ({lam:.4g},{J_sq(Ztr, T9y_tr, b, w):.4f}) ({lam:.4g},{J_sq(Zte, T9y_te, b, w):.4f})")

print("\n[7.4] degree-9 curves at lam = 0.1 and 10 for the figure")
for lam in (0.1, 10):
    b, w = ridge(Ztr, T9y_tr, lam)
    v = b + P9(grid) @ w
    print(f"  L{lam}:", "".join(f"({a:.1f},{c:.3f})" for a, c in zip(grid, v)))

print("\n[7.4] shrinkage factor 1 - alpha*lam/m")
for alpha, lam, m in [(0.01, 1, 10), (0.1, 1, 10), (0.01, 10, 10), (0.1, 10, 50)]:
    print(f"  alpha={alpha}, lam={lam}, m={m}:  factor = {1 - alpha * lam / m:.4f}")

print("\n[7.4] gradient descent agrees with the closed form (degree 9, lam = 1)")
lam, alpha = 1.0, 0.05
b, w = 0.0, np.zeros(9)
m = len(T9y_tr)
for it in range(200000):
    e = b + Ztr @ w - T9y_tr
    w = w * (1 - alpha * lam / m) - alpha * (Ztr.T @ e) / m
    b = b - alpha * e.mean()
bc, wc = ridge(Ztr, T9y_tr, lam)
print(f"  max |w_gd - w_closed| = {np.abs(w - wc).max():.2e},  |b_gd - b_closed| = {abs(b - bc):.2e}")

print("\n[7.4] the penalty makes X^T X + lam*I' invertible: duplicated feature")
Zdup = np.c_[Ztr[:, :1], Ztr[:, :1]]
A = np.c_[np.ones(m), Zdup]
print(f"  rank of A^T A without penalty: {np.linalg.matrix_rank(A.T @ A)} of 3")
b, w = ridge(Zdup, T9y_tr, 1.0)
print(f"  lam = 1: b = {b:.4f}, w = {w}  (the weight splits equally)")
b1, w1 = ridge(Ztr[:, :1], T9y_tr, 0.5)
print(f"  single copy, lam = 0.5: w = {w1[0]:.4f} -> half of it = {w1[0] / 2:.4f}")

print("\n[7.5] why b is not penalised: shift every y by +100")
for pen_b in (False, True):
    for shift in (0, 100):
        y = T9y_tr + shift
        Z = Ztr[:, :2]
        A = np.c_[np.ones(m), Z]
        R = 10.0 * np.eye(3)
        if not pen_b:
            R[0, 0] = 0
        th = np.linalg.solve(A.T @ A + R, A.T @ y)
        print(f"  penalise b = {pen_b!s:5}, shift {shift:3d}:  b = {th[0]:9.4f}"
              f"  w = {th[1:]}  f(5.5) - shift = {th[0] + P9(np.array([5.5]))[0, :2] @ th[1:] - shift:8.4f}")

print("\n[7.6] regularised logistic regression on T7s (separable)")
for lam in [1e-3, 0.01, 0.1, 1, 10]:
    th = logistic_fit(T7x.reshape(-1, 1), T7y, lam)
    print(f"  lam = {lam:>6g}:  b = {th[0]:9.4f}  w = {th[1]:8.4f}"
          f"  boundary = {-th[0] / th[1]:7.4f}  J(unpenalised) = {logistic_J(T7x.reshape(-1, 1), T7y, th):.6f}")

print("\n[7.6] T8 with all monomials up to degree 6 (27 features), standardised")
x1, x2, y8 = T8[:, 0], T8[:, 1], T8[:, 2]
cols = [x1 ** i * x2 ** (k - i) for k in range(1, 7) for i in range(k + 1)]
F = np.vstack(cols).T
F = (F - F.mean(0)) / F.std(0)
print(f"  number of features: {F.shape[1]}")
for lam in [1e-4, 0.1, 1, 10]:
    th = logistic_fit(F, y8, lam, iters=400)
    acc = ((sig(np.c_[np.ones(20), F] @ th) >= 0.5) == y8).mean()
    print(f"  lam = {lam:>6g}:  train accuracy = {acc:.2f}   ||w|| = {np.linalg.norm(th[1:]):10.2f}"
          f"   J(unpenalised) = {logistic_J(F, y8, th):.4f}")

print("\n[7.7] lasso (L1) on degree 9: coordinate descent")


def lasso(Z, y, lam, iters=20000):
    """Minimise (1/2m)||b + Zw - y||^2 + (lam/m)||w||_1 by coordinate descent."""
    m, n = Z.shape
    w = np.zeros(n)
    b = y.mean()
    for _ in range(iters):
        w_old = w.copy()
        for j in range(n):
            r = y - b - Z @ w + Z[:, j] * w[j]
            rho = Z[:, j] @ r / m
            zj = Z[:, j] @ Z[:, j] / m
            w[j] = np.sign(rho) * max(abs(rho) - lam / m, 0.0) / zj
        b = (y - Z @ w).mean()
        if np.abs(w - w_old).max() < 1e-12:
            break
    return b, w


for lam in [0.1, 1, 5, 20]:
    b, w = lasso(Ztr, T9y_tr, lam)
    bR, wR = ridge(Ztr, T9y_tr, lam)
    print(f"  lam = {lam:>4g}:  L1 nonzero = {int((np.abs(w) > 1e-8).sum())} of 9,"
          f"  J_test(L1) = {J_sq(Zte, T9y_te, b, w):.4f};"
          f"  L2 nonzero = {int((np.abs(wR) > 1e-8).sum())},"
          f"  J_test(L2) = {J_sq(Zte, T9y_te, bR, wR):.4f}")
    print(f"        L1 w = {np.round(w, 4)}")
try:
    from sklearn.linear_model import Lasso
    b, w = lasso(Ztr, T9y_tr, 5)
    sk = Lasso(alpha=5 / len(T9y_tr), max_iter=100000, tol=1e-12).fit(Ztr, T9y_tr)
    print(f"  sklearn check at lam = 5: max |w - w_sklearn| = {np.abs(sk.coef_ - w).max():.2e}")
except ImportError:
    pass

print("\n[7.8] pitfall: the same ridge fit in two units (degree 2, lam = 10, unscaled)")
for scale, unit in [(1.0, "metres"), (100.0, "centimetres")]:
    x = T9x_tr * scale
    Z = np.vstack([x, x ** 2]).T
    b, w = ridge(Z, T9y_tr, 10.0)
    pred = b + np.array([5.5 * scale, (5.5 * scale) ** 2]) @ w
    print(f"  x in {unit:11}:  w = {w}  b = {b:.4f}  f(5.5 m) = {pred:.4f}"
          f"  J_train = {J_sq(Z, T9y_tr, b, w):.4f}")

print("\n[7.7] constraint-region picture: Q(w) = (w1-1.6)^2 + (w2-0.6)^2 + 1.2(w1-1.6)(w2-0.6)")
Q = lambda w1, w2: (w1 - 1.6) ** 2 + (w2 - 0.6) ** 2 + 1.2 * (w1 - 1.6) * (w2 - 0.6)
t = np.linspace(0, 2 * np.pi, 2_000_001)
c1, c2 = np.cos(t), np.sin(t)
i = np.argmin(Q(c1, c2))
print(f"  L2 ball (radius 1): optimum w = ({c1[i]:.4f}, {c2[i]:.4f}), Q = {Q(c1[i], c2[i]):.4f}")
s = np.linspace(-1, 1, 2_000_001)
cand = [(s, 1 - abs(s)), (s, -(1 - abs(s)))]
best = min(((Q(a, b).min(), a[np.argmin(Q(a, b))], b[np.argmin(Q(a, b))]) for a, b in cand))
print(f"  L1 ball (radius 1): optimum w = ({best[1]:.4f}, {best[2]:.4f}), Q = {best[0]:.4f}")
print(f"  contour levels through the two optima: {Q(c1[i], c2[i]):.4f}, {best[0]:.4f}")
