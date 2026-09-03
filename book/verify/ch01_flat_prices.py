"""Chapter 1 --- numbers for the end-to-end tour (Section 1.8).

Dataset D1 "city apartments" (Appendix D): x = floor area in hundreds of
square feet, y = sale price in thousands of dollars.  Twelve points.

Run:  /tmp/mlenv/bin/python verify/ch01_flat_prices.py
"""

import numpy as np

X = np.array([5, 7, 9, 10, 12, 13, 15, 16, 18, 20, 22, 24], dtype=float)
Y = np.array([22, 26, 36, 33, 44, 49, 52, 61, 63, 72, 74, 84], dtype=float)
m = X.size


def fit(x, y):
    """Closed-form least squares for y = w x + b."""
    w = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return w, y.mean() - w * x.mean()


def cost(x, y, w, b):
    return ((w * x + b - y) ** 2).sum() / (2 * x.size)


def rmse(x, y, w, b):
    return float(np.sqrt(((w * x + b - y) ** 2).mean()))


print(f"m = {m}, xbar = {X.mean():.4f}, ybar = {Y.mean():.4f}")

w, b = fit(X, Y)
print(f"\n[1.8 full-data fit]  w = {w:.4f}   b = {b:.4f}")
print(f"    J(w,b)         = {cost(X, Y, w, b):.4f}")
print(f"    RMSE           = {rmse(X, Y, w, b):.4f} ")
print(f"    prediction at x=17 (1700 sq ft) = {w * 17 + b:.4f} ")

# Baseline: predict the mean price for every flat.
print(f"\n[1.8 baseline]  constant model  yhat = ybar = {Y.mean():.4f}")
print(f"    J              = {cost(X, Y, 0.0, Y.mean()):.4f}")
print(f"    RMSE           = {rmse(X, Y, 0.0, Y.mean()):.4f} ")

# Held-out check: train on the first 9 flats, test on the last 3.
xtr, ytr, xte, yte = X[:9], Y[:9], X[9:], Y[9:]
wt, bt = fit(xtr, ytr)
print(f"\n[1.8 held-out]  train on 9, test on 3:  w = {wt:.4f}  b = {bt:.4f}")
print(f"    train RMSE     = {rmse(xtr, ytr, wt, bt):.4f} ")
print(f"    test  RMSE     = {rmse(xte, yte, wt, bt):.4f} ")
for xi, yi in zip(xte, yte):
    print(f"      x={xi:5.1f}  y={yi:5.1f}  yhat={wt * xi + bt:6.2f}"
          f"  err={wt * xi + bt - yi:+6.2f}")

# A few costs quoted in the text for badly chosen parameters.
for (ww, bb) in [(3.6, 3.0), (0.0, 0.0), (4.0, 0.0), (3.0, 10.0)]:
    print(f"\n[1.8 cost probe]  J({ww}, {bb}) = {cost(X, Y, ww, bb):.4f}"
          f"   RMSE = {rmse(X, Y, ww, bb):.4f}")

# Largest stable step size for batch gradient descent on this quadratic cost:
# the Hessian of J is (1/m) * [[sum x^2, sum x], [sum x, m]] and gradient
# descent diverges once alpha exceeds 2 / lambda_max.
H = np.array([[(X ** 2).sum(), X.sum()], [X.sum(), float(m)]]) / m
lam = np.linalg.eigvalsh(H)
print(f"\n[1.8 stability]  Hessian eigenvalues = {lam[0]:.4f}, {lam[1]:.4f}")
print(f"    divergence threshold alpha* = 2/lambda_max = {2 / lam.max():.6f}")

# Gradient-descent tour: the run quoted in Section 1.8.
alpha, w_gd, b_gd = 0.005, 0.0, 0.0
print(f"\n[1.8 gradient descent, alpha = {alpha}]")
for k in range(1, 20001):
    e = (w_gd * X + b_gd) - Y
    gw, gb = (e * X).mean(), e.mean()
    w_gd, b_gd = w_gd - alpha * gw, b_gd - alpha * gb
    if k in (1, 10, 100, 1000, 5000, 20000):
        print(f"    iter {k:5d}:  w = {w_gd:8.4f}  b = {b_gd:8.4f}"
              f"   J = {cost(X, Y, w_gd, b_gd):9.4f}")
print(f"    optimum      :  w = {w:8.4f}  b = {b:8.4f}"
      f"   J = {cost(X, Y, w, b):9.4f}")

# ---------------------------------------------------------------------------
# pgfplots coordinates for Figure "cost against iteration" in Section 1.8.
# ---------------------------------------------------------------------------
marks = sorted({int(round(10 ** (k / 8))) for k in range(0, 8 * 5 + 1)})
w_gd, b_gd, trace, it = 0.0, 0.0, [], 0
for target in marks:
    while it < target:
        e = (w_gd * X + b_gd) - Y
        w_gd -= alpha * (e * X).mean()
        b_gd -= alpha * e.mean()
        it += 1
    trace.append((it, cost(X, Y, w_gd, b_gd)))
print("\n[1.8 figure coordinates: (iteration, J)]")
print("    " + " ".join(f"({i},{j:.4f})" for i, j in trace))

# ---------------------------------------------------------------------------
# The two wrong answers quoted in the Lab box of Section 1.8.
# ---------------------------------------------------------------------------
J = cost(X, Y, w, b)
print("\n[1.8 lab box: correct and two common mistakes]")
print(f"    correct  J                        = {J:.4f}")
print(f"    forgot the 1/2  (sum sq / m)      = {2 * J:.4f}")
print(f"    forgot the 1/m  (sum sq / 2)      = {J * m:.4f}")
