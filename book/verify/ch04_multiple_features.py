"""Chapter 4 --- numbers for multiple linear regression and vectorization.

Dataset D2 (Appendix D): five apartments, two features.
  x1 = floor area, hundreds of square feet
  x2 = age of the building, years
  y  = sale price, thousands of dollars

Timings are hardware-dependent; the text says so, and quotes the values this
script produced on the machine used to prepare the book.

Run:  /tmp/mlenv/bin/python verify/ch04_multiple_features.py
"""

import platform
import time

import numpy as np

X2 = np.array([[8.0, 20.0],
               [12.0, 10.0],
               [15.0, 12.0],
               [18.0, 5.0],
               [22.0, 8.0]])
y2 = np.array([30.0, 46.0, 55.0, 72.0, 84.0])
m, n = X2.shape


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def normal_equation(X, y):
    Xb = add_bias(X)
    return np.linalg.solve(Xb.T @ Xb, Xb.T @ y)


def cost(X, y, w, b):
    return float(((X @ w + b - y) ** 2).sum() / (2 * X.shape[0]))


print(f"[machine] {platform.machine()}, numpy {np.__version__}")
print(f"\n[4.1 dataset D2]  m = {m}, n = {n}")
Xb = add_bias(X2)
print("    design matrix with bias column:")
for row, yi in zip(Xb, y2):
    print(f"      [{row[0]:.0f}  {row[1]:5.1f}  {row[2]:5.1f}]   y = {yi:.0f}")

# ------------------------------------------------- 4.5 the normal equation
theta = normal_equation(X2, y2)
b_ne, w_ne = theta[0], theta[1:]
print(f"\n[4.5 normal equation]")
print(f"    X^T X =")
G = Xb.T @ Xb
for r in G:
    print("      [" + "  ".join(f"{v:9.1f}" for v in r) + "]")
print(f"    X^T y = [" + "  ".join(f"{v:9.1f}" for v in Xb.T @ y2) + "]")
print(f"    b  = {b_ne:.6f}")
print(f"    w1 = {w_ne[0]:.6f}   (per hundred sq ft)")
print(f"    w2 = {w_ne[1]:.6f}   (per year of age)")
print(f"    J  = {cost(X2, y2, w_ne, b_ne):.6f}")
print(f"    predictions: " + "  ".join(f"{v:.3f}" for v in X2 @ w_ne + b_ne))
print(f"    residuals  : " + "  ".join(f"{v:+.3f}" for v in X2 @ w_ne + b_ne - y2))

# ------------------------------------------------- 4.4 gradient descent, vector form
def gd(X, y, alpha, iters, w=None, b=0.0):
    mm = X.shape[0]
    w = np.zeros(X.shape[1]) if w is None else w
    for _ in range(iters):
        r = X @ w + b - y
        w, b = w - alpha * (X.T @ r) / mm, b - alpha * r.mean()
    return w, float(b)

# Unscaled features need a tiny step; scaled ones do not (Chapter 5).
mu, sigma = X2.mean(axis=0), X2.std(axis=0)
Xs = (X2 - mu) / sigma
H = (add_bias(Xs).T @ add_bias(Xs)) / m
print(f"\n[4.4 gradient descent in vector form, on z-scored features]")
print(f"    alpha* = 2/lambda_max = {2/np.linalg.eigvalsh(H).max():.4f}")
for iters in [10, 100, 1000, 10000, 100000]:
    ws, bs = gd(Xs, y2, 0.3, iters)
    # map back to the original feature scale
    w_back = ws / sigma
    b_back = bs - float((ws * mu / sigma).sum())
    print(f"    {iters:>6} iters:  b = {b_back:12.6f}"
          f"   w1 = {w_back[0]:.6f}   w2 = {w_back[1]:.6f}"
          f"   J = {cost(X2, y2, w_back, b_back):.6f}")
ws, bs = gd(Xs, y2, 0.3, 100000)
w_back, b_back = ws / sigma, bs - float((ws * mu / sigma).sum())
print(f"    normal equation:  b = {b_ne:12.6f}   w1 = {w_ne[0]:.6f}"
      f"   w2 = {w_ne[1]:.6f}   J = {cost(X2,y2,w_ne,b_ne):.6f}")
print(f"    max |difference| = "
      f"{max(abs(b_back-b_ne), *abs(w_back-w_ne)):.2e}")

# ------------------------------------------------- 4.3 vectorization speed-up
print("\n[4.3 vectorization: dot product of two vectors of length N]")
rng = np.random.default_rng(0)
for N in [1_000, 100_000, 1_000_000]:
    a, bb = rng.random(N), rng.random(N)

    t0 = time.perf_counter()
    total = 0.0
    for i in range(N):
        total += a[i] * bb[i]
    t_loop = time.perf_counter() - t0

    total_v = float(a @ bb)                  # warm up
    reps, t_vec = 20, 1e9
    while True:                              # time at least 50 ms of work
        t0 = time.perf_counter()
        for _ in range(reps):
            total_v = float(a @ bb)
        elapsed = time.perf_counter() - t0
        if elapsed > 0.05:
            t_vec = elapsed / reps
            break
        reps *= 4

    print(f"    N = {N:>9,}:  loop {t_loop*1e3:9.2f} ms |"
          f" numpy {t_vec*1e3:7.4f} ms |"
          f" speed-up {t_loop/t_vec:8.0f}x |"
          f" agree to {abs(total-total_v):.1e}")

# ------------------------------------------------- 4.5 cost of the normal equation
print("\n[4.5 solving the normal equations as n grows; expect x8 per doubling]")
timings = {}
for nn in [200, 400, 800, 1600, 3200]:
    A = rng.random((2 * nn, nn))
    yy = rng.random(2 * nn)
    G, c = A.T @ A, A.T @ yy
    np.linalg.solve(G, c)                    # warm up
    reps = 20
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            np.linalg.solve(G, c)
        elapsed = time.perf_counter() - t0
        if elapsed > 0.05:
            break
        reps *= 4
    t = elapsed / reps
    prev = timings.get(nn // 2)
    ratio = f"   x{t/prev:5.2f} vs n/2" if prev else ""
    timings[nn] = t
    print(f"    n = {nn:>5}:  solve took {t*1e3:8.3f} ms{ratio}")

# ------------------------------------------------- 4.6 a singular X^T X
print("\n[4.6 duplicated feature: X^T X becomes singular]")
Xdup = np.hstack([X2, X2[:, [0]]])          # area repeated
Gd = add_bias(Xdup).T @ add_bias(Xdup)
print(f"    det(X^T X) = {np.linalg.det(Gd):.6e}")
print(f"    rank = {np.linalg.matrix_rank(Gd)} of {Gd.shape[0]}")
sol = np.linalg.lstsq(add_bias(Xdup), y2, rcond=None)[0]
print(f"    lstsq still returns: b = {sol[0]:.4f}, w = "
      + ", ".join(f"{v:.4f}" for v in sol[1:]))
print(f"    and w1 + w3 = {sol[1] + sol[3]:.6f}, versus the single-feature"
      f" w1 = {w_ne[0]:.6f}")

# ------------------------------------------------- 4.7 multicollinearity
print("\n[4.7 near-collinearity: a tiny change in y moves the coefficients]")
X3 = np.column_stack([X2[:, 0], X2[:, 0] * 3.0 + np.array([0.02, -0.01, 0.015,
                                                           -0.02, 0.005])])
cond = np.linalg.cond(add_bias(X3).T @ add_bias(X3))
print(f"    condition number of X^T X = {cond:.3e}")
for label, yy in [("original", y2), ("y[0] + 0.1", y2 + np.array([0.1,0,0,0,0]))]:
    t = np.linalg.solve(add_bias(X3).T @ add_bias(X3), add_bias(X3).T @ yy)
    print(f"    {label:<12}: b = {t[0]:10.3f}  w1 = {t[1]:10.3f}"
          f"  w2 = {t[2]:10.3f}")

# ------------------------------------------------- 4.7 the same data, one feature
print("\n[4.7 for comparison: area alone, no second feature]")
t1 = np.linalg.solve(add_bias(X2[:, [0]]).T @ add_bias(X2[:, [0]]),
                     add_bias(X2[:, [0]]).T @ y2)
print(f"    b = {t1[0]:.6f}   w1 = {t1[1]:.6f}"
      f"   J = {cost(X2[:, [0]], y2, t1[1:], t1[0]):.6f}")
print(f"    versus two features:  J = {cost(X2, y2, w_ne, b_ne):.6f}")
