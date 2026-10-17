"""Chapter 15 --- numbers for recommender systems.

Datasets:
  R54  five dishes x four users of a food-delivery app (shared with the companion
       lecture): ratings 0-5, with five entries missing.  Hand-scored features
       x1 = spicy, x2 = sweet.
  S80  synthetic ratings: 200 users x 80 dishes generated from 3 latent tastes plus
       noise (sd 0.5), 30% of entries observed; observed ratings split 80/20 into
       training and held-out test ratings.

Run:  /tmp/mlenv/bin/python verify/ch15_recommenders.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
nan = np.nan
DISH = ["Chicken Curry", "Chilli Noodles", "Pepper Soup", "Ginger Cake", "Mango Ice Cream"]
USER = ["Alice", "Bob", "Carol", "Dave"]
Y = np.array([[5, 5, 0, 0], [5, nan, nan, 0], [nan, 4, 0, nan], [0, 0, 5, 4], [0, 0, 5, nan]])
R = (~np.isnan(Y)).astype(float)
Y0 = np.nan_to_num(Y)
Xc = np.array([[0.9, 0], [1.0, 0.01], [0.99, 0], [0.1, 1.0], [0, 0.9]])

print("[15.2] content-based prediction: Alice w = (5, 0), b = 0")
print(f"  Pepper Soup x = (0.99, 0): {5 * 0.99 + 0 * 0:.2f}")
print("[15.2] fitted content-based models, lambda = 1 on w")
for j, u in enumerate(USER):
    idx = R[:, j] > 0
    A = np.c_[np.ones(idx.sum()), Xc[idx]]
    Reg = np.eye(3); Reg[0, 0] = 0
    th = np.linalg.solve(A.T @ A + Reg, A.T @ Y0[idx, j])
    print(f"  {u}: b {th[0]:.3f}, w {th[1:].round(3)}, predictions {(np.c_[np.ones(5), Xc] @ th).round(2)}")


def cf_cost_grad(X, W, b, Yn, Rv, lam):
    E = (X @ W.T + b - Yn) * Rv
    J = 0.5 * (E ** 2).sum() + lam / 2 * ((X ** 2).sum() + (W ** 2).sum())
    return J, E @ W + lam * X, E.T @ X + lam * W, E.sum(0)


print("\n[15.3] two gradient steps by hand: n = 1 feature, x = w = 1, b = 0, lambda = 0, alpha = 0.1")
X = np.ones((5, 1)); W = np.ones((4, 1)); b = np.zeros(4)
for step in (1, 2):
    J, gX, gW, gb = cf_cost_grad(X, W, b, Y0, R, 0.0)
    print(f"  step {step}: J = {J:.4f}")
    print(f"    dJ/dx = {gX.ravel().round(4)}\n    dJ/dw = {gW.ravel().round(4)}\n    dJ/db = {gb.round(4)}")
    X, W, b = X - 0.1 * gX, W - 0.1 * gW, b - 0.1 * gb
    print(f"    x = {X.ravel().round(4)}\n    w = {W.ravel().round(4)}\n    b = {b.round(4)}")
J, *_ = cf_cost_grad(X, W, b, Y0, R, 0.0)
print(f"  J after two steps = {J:.4f}")
E0 = (np.ones((5, 4)) - Y0) * R
print(f"  first step, dish 1 by hand: residuals {E0[0]}, dJ/dx1 = sum(e*w) = {E0[0].sum():.1f}")

print("\n[15.3] full training: n = 2, lambda = 0.1, alpha = 0.01, 2000 steps (lecture run)")
r = np.random.default_rng(1)
X = r.normal(0, 0.5, (5, 2)); W = r.normal(0, 0.5, (4, 2)); b = np.zeros(4)
for t in range(2001):
    J, gX, gW, gb = cf_cost_grad(X, W, b, Y0, R, 0.1)
    if t in (0, 40, 80, 2000):
        print(f"  t = {t:4d}: J = {J:.3f}")
    if t < 2000:
        X, W, b = X - 0.01 * gX, W - 0.01 * gW, b - 0.01 * gb
P = X @ W.T + b
print("  predictions:\n", P.round(2))
D = ((X[:, None] - X[None]) ** 2).sum(2)
print("  squared distances from Pepper Soup:", D[2].round(3))
print("  learned X:\n", X.round(2))

print("\n[15.5] mean normalisation and a new user (Eve)")
mu = (Y0 * R).sum(1) / R.sum(1)
print("  dish means:", mu.round(2))

print("\n[15.4] binary labels from the ratings: like = rating >= 4")
print("  ", np.where(R > 0, (Y0 >= 4).astype(int), -1).tolist())

print("\n[15.8] S80: held-out evaluation")
g = np.random.default_rng(15)
nu, nm, k = 200, 80, 3
Ut, Vt = g.normal(0, 1, (nu, k)), g.normal(0, 1, (nm, k))
Ytrue = np.clip(3 + (Vt @ Ut.T) * 0.8 + g.normal(0, 0.5, (nm, nu)), 0, 5)
Obs = g.uniform(size=(nm, nu)) < 0.30
idx = np.argwhere(Obs)
g.shuffle(idx)
ntest = len(idx) // 5
test, train = idx[:ntest], idx[ntest:]
Rtr = np.zeros((nm, nu)); Rtr[train[:, 0], train[:, 1]] = 1
print(f"  observed ratings {len(idx)} ({100 * len(idx) / (nm * nu):.1f}%), train {len(train)}, test {len(test)}")
mu_i = (Ytrue * Rtr).sum(1) / np.maximum(Rtr.sum(1), 1)
glob = (Ytrue * Rtr).sum() / Rtr.sum()
rmse = lambda P: float(np.sqrt(np.mean((P[test[:, 0], test[:, 1]] - Ytrue[test[:, 0], test[:, 1]]) ** 2)))
print(f"  baseline global mean: test RMSE {rmse(np.full((nm, nu), glob)):.4f}")
print(f"  baseline dish means:  test RMSE {rmse(np.repeat(mu_i[:, None], nu, 1)):.4f}")


def train_cf(n, lam, steps=3000, alpha=0.005, seed=0):
    gg = np.random.default_rng(seed)
    X = gg.normal(0, 0.1, (nm, n)); W = gg.normal(0, 0.1, (nu, n)); b = np.zeros(nu)
    Yn = (Ytrue - mu_i[:, None]) * Rtr
    m = np.zeros(1)
    # Adam on all parameters
    ps = [X, W, b]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    for t in range(1, steps + 1):
        J, gX, gW, gb = cf_cost_grad(ps[0], ps[1], ps[2], Yn, Rtr, lam)
        for i, gi in enumerate((gX, gW, gb)):
            ms[i] = 0.9 * ms[i] + 0.1 * gi; vs[i] = 0.999 * vs[i] + 0.001 * gi * gi
            ps[i] = ps[i] - 0.02 * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
    X, W, b = ps
    P = X @ W.T + b + mu_i[:, None]
    tr = float(np.sqrt(np.mean((P[train[:, 0], train[:, 1]] - Ytrue[train[:, 0], train[:, 1]]) ** 2)))
    return tr, rmse(P), P


print("  collaborative filtering (mean-normalised, Adam, 3000 steps):")
for n, lam in [(1, 1.0), (3, 1.0), (3, 10.0), (10, 0.0), (10, 1.0), (10, 10.0)]:
    tr, te, _ = train_cf(n, lam)
    print(f"    n = {n:2d}, lambda = {lam:3.1f}:  train RMSE {tr:.4f}   test RMSE {te:.4f}")

print("\n[15.9] retrieval and ranking arithmetic")
items = 5_000_000; cand = 500; d = 32
print(f"  scoring every item: {items * d:,} multiply-adds per request; candidates only: {cand * d:,}"
      f" ({items // cand:,} times fewer)")
