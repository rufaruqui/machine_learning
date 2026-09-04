"""Chapter 5 --- numbers for feature scaling, convergence and polynomial features.

Datasets:
  D2  five apartments, two features (area, age)     -- from Chapter 4
  T6  six building plots (frontage, depth, price)   -- feature engineering
  T5  ten points on a curve                         -- polynomial regression

Run:  /tmp/mlenv/bin/python verify/ch05_scaling.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)

D2x = np.array([[8.0, 20.0], [12.0, 10.0], [15.0, 12.0],
                [18.0, 5.0], [22.0, 8.0]])
D2y = np.array([30.0, 46.0, 55.0, 72.0, 84.0])


def bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


def cost(X, y, theta):
    return float(((bias(X) @ theta - y) ** 2).sum() / (2 * X.shape[0]))


def hess(X):
    Xb = bias(X)
    return Xb.T @ Xb / X.shape[0]


def report(name, X):
    H = hess(X)
    lam = np.linalg.eigvalsh(H)
    print(f"    {name:<22} lambda_min = {lam[0]:10.5f}  lambda_max = {lam.max():11.4f}"
          f"  cond = {lam.max()/lam[0]:12.1f}  alpha* = {2/lam.max():.6f}")
    return lam


# ------------------------------------------------- 5.2 what scaling does
print("[5.2 feature ranges in D2]")
for j, nm in enumerate(["area", "age"]):
    c = D2x[:, j]
    print(f"    {nm:<6} min {c.min():5.1f}  max {c.max():5.1f}  range {np.ptp(c):5.1f}"
          f"  mean {c.mean():6.2f}  sd {c.std():6.3f}")

print("\n[5.2 conditioning of J under four scalings]")
mu, sd, lo, hi = D2x.mean(0), D2x.std(0), D2x.min(0), D2x.max(0)
scalings = {
    "raw": D2x,
    "divide by max": D2x / hi,
    "mean normalized": (D2x - mu) / (hi - lo),
    "z-scored": (D2x - mu) / sd,
}
for nm, Xs in scalings.items():
    report(nm, Xs)

# ------------------------------------------------- 5.3 iterations needed
def gd(X, y, alpha, target, cap=2_000_000):
    """Iterations until J is within 1% of the minimum, or cap."""
    Xb = bias(X)
    m = X.shape[0]
    theta = np.zeros(Xb.shape[1])
    for k in range(1, cap + 1):
        r = Xb @ theta - y
        theta = theta - alpha * (Xb.T @ r) / m
        if not np.isfinite(theta).all():
            return None, np.inf
        if cost(X, y, theta) <= target:
            return k, cost(X, y, theta)
    return None, cost(X, y, theta)


Jmin = cost(D2x, D2y, np.linalg.solve(bias(D2x).T @ bias(D2x),
                                      bias(D2x).T @ D2y))
target = Jmin * 1.01
print(f"\n[5.3 iterations to reach J <= 1.01 * Jmin = {target:.6f}"
      f"   (Jmin = {Jmin:.6f})]")
for nm, Xs in scalings.items():
    lam = np.linalg.eigvalsh(hess(Xs))
    a = 0.9 * 2 / lam.max()          # a safe 90% of the stability limit
    k, J = gd(Xs, D2y, a, target)
    print(f"    {nm:<22} alpha = {a:.6f}   iterations = "
          f"{k if k else '> 2,000,000':>11}   J = {J:.6f}")

# ------------------------------------------------- 5.4 the alpha ladder
print("\n[5.4 the x3 ladder on raw features, 200 iterations each]")
for a in [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    Xb, theta = bias(D2x), np.zeros(3)
    for _ in range(200):
        theta = theta - a * (Xb.T @ (Xb @ theta - D2y)) / 5
    J = cost(D2x, D2y, theta)
    flag = "diverged" if not np.isfinite(J) or J > 1e6 else f"J = {J:12.4f}"
    print(f"    alpha = {a:<8g} {flag}")

print("\n[5.4 the same ladder on z-scored features]")
Xz = scalings["z-scored"]
for a in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
    Xb, theta = bias(Xz), np.zeros(3)
    for _ in range(200):
        theta = theta - a * (Xb.T @ (Xb @ theta - D2y)) / 5
    J = cost(Xz, D2y, theta)
    flag = "diverged" if not np.isfinite(J) or J > 1e6 else f"J = {J:12.6f}"
    print(f"    alpha = {a:<8g} {flag}")

# ------------------------------------------------- 5.5 scaling statistics leak
print("\n[5.5 does the choice of scaling statistics change the model?]")
rng = np.random.default_rng(3)
n_tr, n_te = 60, 20
Xall = np.column_stack([rng.uniform(5, 25, n_tr + n_te),
                        rng.uniform(0, 30, n_tr + n_te)])
yall = 3.0 * Xall[:, 0] - 0.4 * Xall[:, 1] + 4.0 + rng.normal(0, 1.5,
                                                              n_tr + n_te)
Xtr, ytr, Xte, yte = Xall[:n_tr], yall[:n_tr], Xall[n_tr:], yall[n_tr:]

mu_tr, sd_tr = Xtr.mean(0), Xtr.std(0)
mu_all, sd_all = Xall.mean(0), Xall.std(0)
print(f"    train-only stats: mean {np.round(mu_tr,4)}  sd {np.round(sd_tr,4)}")
print(f"    all-data   stats: mean {np.round(mu_all,4)}  sd {np.round(sd_all,4)}")


def preds(mu_, sd_, lam=0.0):
    Z = bias((Xtr - mu_) / sd_)
    P = lam * np.eye(Z.shape[1])
    P[0, 0] = 0.0                       # never penalise the bias
    th = np.linalg.solve(Z.T @ Z + P, Z.T @ ytr)
    return bias((Xte - mu_) / sd_) @ th


def rmse(p):
    return float(np.sqrt(((p - yte) ** 2).mean()))


p_tr, p_all = preds(mu_tr, sd_tr), preds(mu_all, sd_all)
print(f"    ordinary least squares, fitted to convergence:")
print(f"      train stats  test RMSE = {rmse(p_tr):.6f}")
print(f"      all   stats  test RMSE = {rmse(p_all):.6f}")
print(f"      max |difference| between the two prediction vectors ="
      f" {np.abs(p_tr - p_all).max():.3e}")

for lam in [1.0, 10.0]:
    q_tr, q_all = preds(mu_tr, sd_tr, lam), preds(mu_all, sd_all, lam)
    print(f"    ridge, lambda = {lam:g} (Chapter 7):")
    print(f"      train stats  test RMSE = {rmse(q_tr):.6f}")
    print(f"      all   stats  test RMSE = {rmse(q_all):.6f}")
    print(f"      max |difference| = {np.abs(q_tr - q_all).max():.6f}")

# And with a fixed iteration budget, the two disagree even without a penalty.
def gd_budget(mu_, sd_, alpha, iters):
    Z = bias((Xtr - mu_) / sd_)
    th = np.zeros(Z.shape[1])
    for _ in range(iters):
        th = th - alpha * (Z.T @ (Z @ th - ytr)) / n_tr
    return bias((Xte - mu_) / sd_) @ th


for iters in [20, 200]:
    a_, b_ = gd_budget(mu_tr, sd_tr, 0.3, iters), gd_budget(mu_all, sd_all, 0.3, iters)
    print(f"    gradient descent stopped after {iters:>3} iterations:"
          f"  RMSE {rmse(a_):.6f} vs {rmse(b_):.6f}"
          f"   (max diff {np.abs(a_ - b_).max():.4f})")

# ------------------------------------------------- 5.6 feature engineering
print("\n[5.6 feature engineering: frontage x depth]")
f = np.array([20.0, 30.0, 25.0, 40.0, 35.0, 45.0])
d = np.array([50.0, 40.0, 60.0, 30.0, 45.0, 50.0])
area = f * d
price = np.array([51.0, 59.0, 76.0, 61.0, 78.0, 113.0])
print("    frontage depth   area  price")
for a_, b_, c_, p_ in zip(f, d, area, price):
    print(f"    {a_:8.0f} {b_:5.0f} {c_:6.0f} {p_:6.0f}")
for nm, X in [("frontage, depth", np.column_stack([f, d])),
              ("area alone", area[:, None]),
              ("frontage, depth, area", np.column_stack([f, d, area]))]:
    th = np.linalg.lstsq(bias(X), price, rcond=None)[0]
    print(f"    {nm:<24} J = {cost(X, price, th):9.4f}"
          f"   theta = {np.round(th, 4)}")

# ------------------------------------------------- 5.7 polynomial regression
print("\n[5.7 polynomial regression on T5]")
px = np.arange(1.0, 11.0)
py = np.array([2.1, 3.4, 5.6, 9.1, 13.2, 19.4, 25.9, 34.2, 43.6, 54.5])
print("    x: " + " ".join(f"{v:.0f}" for v in px))
print("    y: " + " ".join(f"{v:.1f}" for v in py))
for deg in [1, 2, 3, 4, 9]:
    V = np.column_stack([px ** k for k in range(1, deg + 1)])
    th = np.linalg.lstsq(bias(V), py, rcond=None)[0]
    cond = np.linalg.cond(bias(V).T @ bias(V))
    print(f"    degree {deg}:  J = {cost(V, py, th):12.6f}"
          f"   cond(X^T X) = {cond:10.3e}")

print("\n[5.7 what powers do to the feature ranges]")
for k in range(1, 5):
    col = px ** k
    print(f"    x^{k}: min {col.min():10.1f}  max {col.max():12.1f}"
          f"  ratio max/min = {col.max()/col.min():10.1f}")

print("\n[5.7 degree 9 through 10 points: an exact fit that misbehaves]")
V9 = np.column_stack([px ** k for k in range(1, 10)])
th9 = np.linalg.lstsq(bias(V9), py, rcond=None)[0]
V2 = np.column_stack([px ** k for k in range(1, 3)])
th2 = np.linalg.lstsq(bias(V2), py, rcond=None)[0]
for xq in [1.5, 5.5, 9.5, 10.5]:
    p9 = float(np.polyval(th9[::-1], xq))
    p2 = float(th2[0] + th2[1] * xq + th2[2] * xq ** 2)
    print(f"    x = {xq:4.1f}:  degree 2 predicts {p2:10.3f}"
          f"   degree 9 predicts {p9:14.3f}")
print(f"    degree 2 coefficients: {np.round(th2, 4)}")
print(f"    largest degree-9 coefficient magnitude: {np.abs(th9).max():.4e}")

# ------------------------------------------------- 5.2 figure: T4 before/after
print("\n[5.2 the Chapter 2 four-point set, raw and standardised]")
T4x = np.array([1.0, 2.0, 3.0, 4.0])
T4y = np.array([1.0, 3.0, 4.0, 6.0])
mu4, sd4 = T4x.mean(), T4x.std()
print(f"    mean = {mu4}, sd = {sd4:.6f}")

for nm, xx in [("raw", T4x), ("z-scored", (T4x - mu4) / sd4)]:
    m4 = 4
    Sx, Sxx, Sy, Sxy, Syy = (xx.sum(), (xx**2).sum(), T4y.sum(),
                             (xx*T4y).sum(), (T4y**2).sum())
    H = np.array([[Sxx, Sx], [Sx, float(m4)]]) / m4
    lam = np.linalg.eigvalsh(H)
    w = ((xx - xx.mean()) * (T4y - T4y.mean())).sum() / ((xx - xx.mean())**2).sum()
    b = T4y.mean() - w * xx.mean()
    Jm = float(((w*xx + b - T4y)**2).sum() / (2*m4))
    print(f"    {nm:<10} J(w,b) = ({Sxx:.4f}w^2 + {2*Sx:.4f}wb + {m4}b^2"
          f" - {2*Sxy:.4f}w - {2*Sy:.4f}b + {Syy:.4f}) / {2*m4}")
    print(f"    {'':<10} eigenvalues {lam[0]:.4f}, {lam[1]:.4f}"
          f"   cond {lam.max()/lam[0]:8.2f}   alpha* {2/lam.max():.4f}")
    print(f"    {'':<10} optimum w = {w:.6f}  b = {b:.6f}  J = {Jm:.6f}")

    # iterations to reach J <= 1.001 * Jmin, at 90% of the stability limit
    for a in [0.9 * 2/lam.max(), 1.0/lam.max()]:
        ww, bb, k = 0.0, 0.0, 0
        while k < 200000:
            r = ww*xx + bb - T4y
            ww, bb, k = ww - a*(r*xx).mean(), bb - a*r.mean(), k+1
            if float((r**2).sum()/(2*m4)) <= Jm * 1.001:
                break
        print(f"    {'':<10} alpha = {a:.4f}: {k} iterations to J <= 1.001 Jmin")

    # descent path for the figure, at the principled choice alpha = 1/lambda_max
    a = 1.0/lam.max()
    ww, bb, path = 0.0, 0.0, [(0.0, 0.0)]
    for _ in range(40):
        r = ww*xx + bb - T4y
        ww, bb = ww - a*(r*xx).mean(), bb - a*r.mean()
        path.append((ww, bb))
    keep = [0,1,2,3,4,5,6,8,10,13,16,20,25,30,40]
    print(f"    {'':<10} alpha used for the path: {a:.4f}")
    print(f"    {'':<10} path: "
          + " ".join(f"({path[i][0]:.3f},{path[i][1]:.3f},0)" for i in keep))

# ------------------------------------------------- 5.8 figure: polynomial curves
print("\n[5.8 polynomial fits for the figure]")
px = np.arange(1.0, 11.0)
py = np.array([2.1, 3.4, 5.6, 9.1, 13.2, 19.4, 25.9, 34.2, 43.6, 54.5])
for deg in [1, 2, 9]:
    V = np.column_stack([px ** k for k in range(1, deg + 1)])
    th = np.linalg.lstsq(bias(V), py, rcond=None)[0]
    print(f"    degree {deg} coefficients (const first): {np.round(th, 6)}")
    grid = np.linspace(0.6, 10.6, 41)
    vals = np.polyval(th[::-1], grid)
    print(f"      curve: " + " ".join(f"({g:.2f},{v:.3f})"
                                      for g, v in zip(grid, vals)))
