"""Chapter 6 --- numbers for logistic regression.

Datasets:
  T7  eight tumours, one feature (size in cm), binary label
  T8  twenty tumours, two features (size, texture score), binary label

Run:  /tmp/mlenv/bin/python verify/ch06_logistic.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)


# --------------------------------------------------------------- datasets
T7x = np.arange(1.0, 9.0)
T7y = np.array([0.0, 1, 0, 0, 0, 1, 1, 1])

T8 = np.array([
    [2.4, 3.4, 0], [1.8, 0.4, 0], [2.2, 4.6, 0], [2.9, 2.7, 0],
    [3.8, 5.9, 0], [2.0, 3.2, 0], [2.9, 3.1, 0], [2.4, 5.0, 0],
    [3.0, 2.7, 0], [1.9, 2.7, 0], [3.8, 6.3, 1], [3.5, 7.2, 1],
    [3.8, 6.2, 1], [2.4, 4.4, 1], [3.3, 6.7, 1], [3.2, 8.2, 1],
    [3.9, 6.5, 1], [3.9, 6.8, 1], [4.4, 5.5, 1], [4.9, 5.2, 1],
])


# --------------------------------------------------------------- machinery
def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def design(x):
    x = np.atleast_2d(np.asarray(x, float).T).T
    return np.hstack([np.ones((x.shape[0], 1)), x])


def Jlog(X, y, th):
    f = np.clip(sig(X @ th), 1e-300, 1 - 1e-16)
    return float(-(y * np.log(f) + (1 - y) * np.log(1 - f)).mean())


def glog(X, y, th):
    return X.T @ (sig(X @ th) - y) / len(y)


def Hlog(X, y, th):
    d = sig(X @ th) * (1 - sig(X @ th))
    return X.T @ (X * d[:, None]) / len(y)


def Jsq(X, y, th):
    f = sig(X @ th)
    return float(((f - y) ** 2).sum() / (2 * len(y)))


def gsq(X, y, th):
    f = sig(X @ th)
    return X.T @ ((f - y) * f * (1 - f)) / len(y)


def Hsq(X, y, th):
    f = sig(X @ th)
    d = f * (1 - f)
    wt = d * d + (f - y) * d * (1 - 2 * f)
    return X.T @ (X * wt[:, None]) / len(y)


def newton(X, y, iters=200):
    th = np.zeros(X.shape[1])
    for _ in range(iters):
        th = th - np.linalg.solve(Hlog(X, y, th) + 1e-12 * np.eye(len(th)),
                                  glog(X, y, th))
    return th


def gd(X, y, th, alpha, iters, grad):
    th = th.astype(float).copy()
    for _ in range(iters):
        th = th - alpha * grad(X, y, th)
    return th


def lstsq(x, y):
    X = design(x)
    return np.linalg.lstsq(X, y, rcond=None)[0]


X7 = design(T7x)


# ------------------------------------------------- 6.1 linear regression fails
print("[6.1 least squares on a 0/1 target, dataset T7]")
b, w = lstsq(T7x, T7y)
print(f"    fit           b = {b:.6f}   w = {w:.6f}")
print(f"    fitted values {np.round(b + w * T7x, 4)}")
print(f"    boundary (f = 0.5) at x = {(0.5 - b) / w:.4f}")
print(f"    extrapolation f(0) = {b:.4f}   f(12) = {b + 12 * w:.4f}"
      f"   f(25) = {b + 25 * w:.4f}")

print("\n[6.1 one far-away malignant example moves the least-squares boundary]")
print(f"    {'extra x':>9} {'LS b':>10} {'LS w':>9} {'LS boundary':>12}"
      f" {'logistic boundary':>19}")
for extra in [None, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]:
    if extra is None:
        xx, yy = T7x, T7y
    else:
        xx, yy = np.r_[T7x, extra], np.r_[T7y, 1.0]
    bb, ww = lstsq(xx, yy)
    tb, tw = newton(design(xx), yy)
    print(f"    {'none' if extra is None else f'{extra:.0f}':>9}"
          f" {bb:10.6f} {ww:9.6f} {(0.5 - bb) / ww:12.4f} {-tb / tw:19.4f}")
bb, ww = lstsq(np.r_[T7x, 20.0], np.r_[T7y, 1.0])
print(f"    with the point at x = 20, the eight fitted values compress to")
print(f"    {np.round(bb + ww * T7x, 4)}")
tb, tw = newton(design(np.r_[T7x, 20.0]), np.r_[T7y, 1.0])
print(f"    logistic fit with that point: b = {tb:.6f}  w = {tw:.6f}"
      f"  boundary = {-tb / tw:.4f}")


# ------------------------------------------------------------- 6.2 sigmoid
print("\n[6.2 the sigmoid and its derivative]")
for z in [-6, -4, -2, -1, 0, 1, 2, 4, 6]:
    g = sig(z)
    print(f"    z = {z:>3}   g(z) = {g:.6f}   g'(z) = {g * (1 - g):.6f}")
print(f"    g(0) = {sig(0):.4f},  max g' = {0.25:.4f} at z = 0")


# ------------------------------------------- 6.4 why squared error fails
print("\n[6.4 the squared-error cost on the sigmoid model is not convex]")
for th in [np.array([0.0, 0.0]), np.array([4.0, -1.0]), np.array([2.0, -0.8])]:
    e = np.linalg.eigvalsh(Hsq(X7, T7y, th))
    print(f"    (b, w) = ({th[0]:5.2f}, {th[1]:5.2f})  J = {Jsq(X7, T7y, th):.6f}"
          f"   eigenvalues = ({e[0]:+.6f}, {e[1]:+.6f})")
mn = 1e9
for bb in np.linspace(-10, 10, 201):
    for ww in np.linspace(-5, 5, 201):
        mn = min(mn, np.linalg.eigvalsh(Hlog(X7, T7y, np.array([bb, ww])))[0])
print(f"    smallest logistic-Hessian eigenvalue over a 201x201 grid: {mn:.3e}")

print("\n[6.4 the plateau: slice at b = -2.673380]")
bo = -2.673380
for ww in [-2.5, -2.0, -1.0, 0.0, 0.5, 0.594084, 1.0, 2.0, 3.0]:
    th = np.array([bo, ww])
    print(f"    w = {ww:6.3f}   J_sq = {Jsq(X7, T7y, th):.6f}"
          f"   J_log = {Jlog(X7, T7y, th):.6f}"
          f"   |grad J_sq| = {np.linalg.norm(gsq(X7, T7y, th)):.6f}")

print("\n[6.4 figure coordinates: slice at b = -2.673380]")
for name, JJ, scale in [("Jsq", Jsq, 1.0), ("Jlog", Jlog, 1.0)]:
    pts = " ".join(f"({ww:.2f},{JJ(X7, T7y, np.array([bo, ww])) * scale:.4f})"
                   for ww in np.arange(-2.5, 3.001, 0.25))
    print(f"    {name}: {pts}")

print("\n[6.4 gradient descent from (b, w) = (4, -1), alpha = 0.2]")
print(f"    {'iter':>7} {'squared-error J':>16} {'(b, w)':>22}"
      f" {'log-loss J':>12} {'(b, w)':>22}")
ths, thl = np.array([4.0, -1.0]), np.array([4.0, -1.0])
prev = 0
for k in [1, 10, 100, 1000, 10000, 20000]:
    ths = gd(X7, T7y, ths, 0.2, k - prev, gsq)
    thl = gd(X7, T7y, thl, 0.2, k - prev, glog)
    prev = k
    print(f"    {k:>7} {Jsq(X7, T7y, ths):16.6f}"
          f" {f'({ths[0]:+.4f}, {ths[1]:+.4f})':>22}"
          f" {Jlog(X7, T7y, thl):12.6f}"
          f" {f'({thl[0]:+.4f}, {thl[1]:+.4f})':>22}")
ths = gd(X7, T7y, np.array([4.0, -1.0]), 0.2, 200000, gsq)
print(f"    after 200,000 iterations the squared-error run is at "
      f"(b, w) = ({ths[0]:.4f}, {ths[1]:.4f}),")
print(f"    J = {Jsq(X7, T7y, ths):.6f}, gradient norm "
      f"{np.linalg.norm(gsq(X7, T7y, ths)):.2e} --- still not converged.")


# --------------------------------------------------- 6.5 the logistic loss
print("\n[6.5 the two losses on a confidently wrong prediction (y = 1)]")
print(f"    {'f':>7} {'squared loss':>13} {'log loss':>10}"
      f" {'d(sq)/dz':>12} {'d(log)/dz':>11} {'ratio':>8}")
for f in [0.9, 0.5, 0.1, 0.01, 0.001]:
    dsq = (f - 1) * f * (1 - f)
    print(f"    {f:>7} {0.5 * (f - 1) ** 2:13.6f} {-np.log(f):10.6f}"
          f" {dsq:12.8f} {f - 1:11.6f} {(f - 1) / dsq:8.1f}")
print(f"    J at the origin (every f = 1/2) = ln 2 = {np.log(2):.6f}")


# ------------------------------------------- 6.7 gradient descent on T7
print("\n[6.7 gradient descent on T7 by hand, alpha = 0.2]")
th = np.zeros(2)
for k in range(4):
    f = sig(X7 @ th)
    g = glog(X7, T7y, th)
    print(f"    k = {k}: b = {th[0]:9.6f}  w = {th[1]:9.6f}  J = {Jlog(X7, T7y, th):.6f}"
          f"  dJ/db = {g[0]:9.6f}  dJ/dw = {g[1]:9.6f}")
    if k == 0:
        print(f"      per-example at k = 0: f = {np.round(f, 6)}")
        print(f"      (f - y)   = {np.round(f - T7y, 6)}")
        print(f"      (f - y)x  = {np.round((f - T7y) * T7x, 6)}"
              f"   sum = {((f - T7y) * T7x).sum():.6f}")
    if k == 1:
        print(f"      per-example at k = 1: f = {np.round(f, 6)}")
        print(f"      (f - y)   = {np.round(f - T7y, 6)}")
        print(f"      (f - y)x  = {np.round((f - T7y) * T7x, 6)}"
              f"   sum = {((f - T7y) * T7x).sum():.6f}")
    th = th - 0.2 * g

print("\n[6.7 the converged fit on T7]")
th7 = newton(X7, T7y)
p7 = sig(X7 @ th7)
print(f"    b = {th7[0]:.6f}   w = {th7[1]:.6f}   J = {Jlog(X7, T7y, th7):.6f}")
print(f"    decision boundary  x* = -b/w = {-th7[0] / th7[1]:.4f}")
print(f"    odds multiplier per cm  e^w = {np.exp(th7[1]):.4f}")
print(f"    predicted probabilities {np.round(p7, 4)}")
print(f"    accuracy at threshold 0.5 = {((p7 >= 0.5) == T7y).mean():.4f}")

print("\n[6.7 conditioning of the logistic cost at the origin, T7]")
mu, sd = T7x.mean(), T7x.std()
print(f"    mu = {mu:.4f}   sigma = {sd:.6f}")
Xz = design((T7x - mu) / sd)
for name, Xs in [("raw", X7), ("standardised", Xz)]:
    e = np.linalg.eigvalsh(Hlog(Xs, T7y, np.zeros(2)))
    alpha = 0.9 * 2 / e[1]
    th, k = np.zeros(2), 0
    while k < 2_000_000:
        th = th - alpha * glog(Xs, T7y, th)
        k += 1
        if np.linalg.norm(glog(Xs, T7y, th)) < 1e-6:
            break
    print(f"    {name:<13} lambda_min = {e[0]:.5f}  lambda_max = {e[1]:.5f}"
          f"  kappa = {e[1] / e[0]:8.1f}  alpha* = {2 / e[1]:.4f}"
          f"  alpha = {alpha:.4f}  iterations = {k:>5}")
    if name == "standardised":
        print(f"      standardised optimum (b, w) = ({th[0]:.6f}, {th[1]:.6f});"
              f" in raw units w = {th[1] / sd:.6f},"
              f" b = {th[0] - th[1] * mu / sd:.6f}")


print("\n[6.7 separable data: T7 with the 2 cm malignancy removed]")
xs = np.array([1.0, 3, 4, 5, 6, 7, 8])
ys = np.array([0.0, 0, 0, 0, 1, 1, 1])
Xs = design(xs)
th = np.zeros(2)
for k in range(1, 200001):
    th = th - 0.2 * glog(Xs, ys, th)
    if k in (1000, 10000, 100000, 200000):
        print(f"    {k:>7} iterations: b = {th[0]:9.4f}  w = {th[1]:8.4f}"
              f"  boundary = {-th[0] / th[1]:.4f}  J = {Jlog(Xs, ys, th):.6f}")


# ------------------------------------------------------- 6.3 / 6.6 dataset T8
print("\n[6.3 dataset T8: two features, twenty examples]")
X8r, y8 = T8[:, :2], T8[:, 2]
X8 = design(X8r)
th8 = newton(X8, y8)
p8 = sig(X8 @ th8)
print(f"    b = {th8[0]:.6f}   w1 = {th8[1]:.6f}   w2 = {th8[2]:.6f}")
print(f"    J = {Jlog(X8, y8, th8):.6f}   accuracy = {((p8 >= 0.5) == y8).mean():.4f}")
print(f"    boundary  x2 = {-th8[0] / th8[2]:.4f} - {th8[1] / th8[2]:.4f} x1")
for x1 in [1.5, 3.0, 5.0]:
    print(f"      x1 = {x1}:  x2 = {(-th8[0] - th8[1] * x1) / th8[2]:.4f}")
print(f"    probabilities {np.round(p8, 3)}")
wrong = np.where((p8 >= 0.5) != y8.astype(bool))[0]
for i in wrong:
    print(f"    misclassified: x = ({X8r[i, 0]}, {X8r[i, 1]}), y = {int(y8[i])},"
          f" f = {p8[i]:.4f}")


# ------------------------------------------------------- 6.8 the threshold
print("\n[6.8 threshold sweep on T7]")
print(f"    {'t':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}"
      f" {'accuracy':>9} {'precision':>10} {'recall':>8}")
for t in [0.10, 0.15, 0.30, 0.43, 0.50, 0.60, 0.70, 0.85, 0.90]:
    pred = p7 >= t
    TP = int((pred & (T7y == 1)).sum())
    FP = int((pred & (T7y == 0)).sum())
    FN = int((~pred & (T7y == 1)).sum())
    TN = int((~pred & (T7y == 0)).sum())
    prec = TP / (TP + FP) if TP + FP else float("nan")
    print(f"    {t:6.2f} {TP:3d} {FP:3d} {FN:3d} {TN:3d}"
          f" {(TP + TN) / len(T7y):9.3f} {prec:10.3f} {TP / (TP + FN):8.3f}")
print("    cost-weighted choice: one missed malignancy costs ten false alarms")
ts = np.arange(0.0005, 1.0, 0.0005)
costs = []
for t in ts:
    pred = p7 >= t
    FP = int((pred & (T7y == 0)).sum())
    FN = int((~pred & (T7y == 1)).sum())
    costs.append(10 * FN + FP)
costs = np.array(costs)
opt = ts[costs == costs.min()]
print(f"    10*FN + FP is minimised (cost {costs.min()}) for every"
      f" t in ({opt.min():.4f}, {opt.max():.4f}],")
print(f"    i.e. anywhere between f(x=1) = {p7[0]:.4f} and f(x=2) = {p7[1]:.4f};"
      f" t = 0.15 is in that range.")
