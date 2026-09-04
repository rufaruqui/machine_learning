"""Chapter 3 --- numbers for the gradient descent chapter.

Datasets, both introduced in Chapter 2:
  T4  (1,1), (2,3), (3,4), (4,6)   -- the running example; optimum (1.6, -0.5)
  T3  (1,1), (2,2), (3,3)          -- used for the one-parameter picture

Run:  /tmp/mlenv/bin/python verify/ch03_gradient_descent.py
"""

import numpy as np

T4x = np.array([1.0, 2.0, 3.0, 4.0]); T4y = np.array([1.0, 3.0, 4.0, 6.0])
T3x = np.array([1.0, 2.0, 3.0]);      T3y = np.array([1.0, 2.0, 3.0])


def cost(x, y, w, b):
    return float(((w * x + b - y) ** 2).sum() / (2 * x.size))


def grads(x, y, w, b):
    r = w * x + b - y
    return float((r * x).mean()), float(r.mean())


def run(x, y, alpha, n, w=0.0, b=0.0):
    out = [(0, w, b, cost(x, y, w, b))]
    for k in range(1, n + 1):
        gw, gb = grads(x, y, w, b)
        w, b = w - alpha * gw, b - alpha * gb
        out.append((k, w, b, cost(x, y, w, b)))
    return out


# ------------------------------------------------- 3.4 the divergence threshold
for name, x in [("T4", T4x), ("T3", T3x)]:
    m = x.size
    H = np.array([[(x ** 2).sum(), x.sum()], [x.sum(), float(m)]]) / m
    lam = np.linalg.eigvalsh(H)
    print(f"[3.4 stability, {name}]  Hessian = [[{H[0,0]:.4f}, {H[0,1]:.4f}],"
          f" [{H[1,0]:.4f}, {H[1,1]:.4f}]]")
    print(f"    eigenvalues {lam[0]:.4f}, {lam[1]:.4f};"
          f"  alpha* = 2/lambda_max = {2/lam.max():.4f}")

# ------------------------------------------------- 3.5 gradients at the origin
gw, gb = grads(T4x, T4y, 0.0, 0.0)
print(f"\n[3.5 gradients at (0,0) on T4]  dJ/dw = {gw:.4f}   dJ/db = {gb:.4f}")
print(f"    with alpha = 0.1 the first step lands at"
      f" w = {-0.1*gw:.4f}, b = {-0.1*gb:.4f}")

# ------------------------------------------------- 3.6 the iteration table
print("\n[3.6 first five iterations on T4, alpha = 0.1, from (0,0)]")
print(f"    {'k':>2}  {'w':>8}  {'b':>8}  {'dJ/dw':>9}  {'dJ/db':>9}  {'J':>9}")
w, b = 0.0, 0.0
print(f"    {0:>2}  {w:8.4f}  {b:8.4f}  {'':>9}  {'':>9}  {cost(T4x,T4y,w,b):9.4f}")
for k in range(1, 6):
    gw, gb = grads(T4x, T4y, w, b)
    w, b = w - 0.1 * gw, b - 0.1 * gb
    print(f"    {k:>2}  {w:8.4f}  {b:8.4f}  {gw:9.4f}  {gb:9.4f}"
          f"  {cost(T4x,T4y,w,b):9.4f}")

for k in [10, 25, 50, 100, 500, 2000]:
    t = run(T4x, T4y, 0.1, k)[-1]
    print(f"    {t[0]:>4}  w = {t[1]:.4f}  b = {t[2]:.4f}  J = {t[3]:.6f}")
print("    optimum   w = 1.6000  b = -0.5000  J = 0.025000")

# ------------------------------------------------- 3.4 learning-rate sweep
print("\n[3.4 the same run at four step sizes, J after k iterations]")
alphas = [0.01, 0.1, 0.22, 0.25]
print("    {:>6} ".format("k") + "".join(f"{a:>14}" for a in alphas))
traces = {a: run(T4x, T4y, a, 200) for a in alphas}
for k in [1, 5, 10, 25, 50, 100, 200]:
    row = f"    {k:>6} "
    for a in alphas:
        J = traces[a][k][3]
        row += f"{J:14.4f}" if abs(J) < 1e6 else f"{J:14.3e}"
    print(row)

# ------------------------------------------------- 3.7 non-simultaneous update
print("\n[3.7 the non-simultaneous update bug, alpha = 0.1, three steps]")
w_ok, b_ok = 0.0, 0.0
w_bad, b_bad = 0.0, 0.0
for k in range(1, 4):
    gw, gb = grads(T4x, T4y, w_ok, b_ok)
    w_ok, b_ok = w_ok - 0.1 * gw, b_ok - 0.1 * gb
    # wrong: w is updated first, then b's gradient uses the NEW w
    gw2, _ = grads(T4x, T4y, w_bad, b_bad)
    w_bad = w_bad - 0.1 * gw2
    _, gb2 = grads(T4x, T4y, w_bad, b_bad)
    b_bad = b_bad - 0.1 * gb2
    print(f"    step {k}:  correct (w,b) = ({w_ok:.4f}, {b_ok:.4f})"
          f"  J = {cost(T4x,T4y,w_ok,b_ok):.4f}   |   "
          f"wrong ({w_bad:.4f}, {b_bad:.4f})  J = {cost(T4x,T4y,w_bad,b_bad):.4f}")

# ------------------------------------------------- figures
print("\n[3.x figure: descent path on the Chapter 2 contour map, alpha = 0.1]")
path = run(T4x, T4y, 0.1, 60)
pts = [path[i] for i in [0,1,2,3,4,5,6,8,10,13,16,20,25,30,40,50,60]]
print("    " + " ".join(f"({w:.3f},{b:.3f},0)" for _, w, b, _ in pts))

print("\n[3.x figure: J against iteration for three step sizes]")
for a in [0.01, 0.1, 0.22]:
    tr = run(T4x, T4y, a, 120)
    ks = [1,2,3,4,6,8,11,15,20,27,36,48,64,85,120]
    print(f"    alpha = {a}:  " + " ".join(f"({k},{tr[k][3]:.4f})" for k in ks))

print("\n[3.x figure: divergence at alpha = 0.25]")
tr = run(T4x, T4y, 0.25, 40)
print("    " + " ".join(f"({k},{tr[k][3]:.4g})" for k in range(0, 21, 2)))

# ------------------------------------------------- 3.9 stochastic gradient descent
print("\n[3.9 stochastic gradient descent on T4, alpha = 0.05, one epoch = 4 steps]")
rng = np.random.default_rng(7)
w, b, sgd = 0.0, 0.0, [(0, 0.0, 0.0, cost(T4x, T4y, 0.0, 0.0))]
step = 0
for epoch in range(15):
    for i in rng.permutation(4):
        r = w * T4x[i] + b - T4y[i]
        w, b = w - 0.05 * r * T4x[i], b - 0.05 * r
        step += 1
        sgd.append((step, w, b, cost(T4x, T4y, w, b)))
print(f"    after {step} single-example steps:"
      f"  w = {w:.4f}  b = {b:.4f}  J = {cost(T4x,T4y,w,b):.4f}")
print("    path: " + " ".join(f"({w:.3f},{b:.3f},0)" for _, w, b, _ in sgd[:33:2]))

# ------------------------------------------------- 3.2 one-dimensional picture
print("\n[3.2 T3 with b fixed at 0:  J(w) = (14/6)(w-1)^2,  J'(w) = (14/3)(w-1)]")
for w0 in [0.0, 2.0, 1.6]:
    dJ = 14 / 3 * (w0 - 1)
    print(f"    w = {w0:.2f}:  J = {14/6*(w0-1)**2:.4f}   J' = {dJ:+.4f}"
          f"   step with alpha=0.1 -> w = {w0 - 0.1*dJ:.4f}")

# ------------------------------------------------- 3.8 a stopping rule that lies
print("\n[3.8 stopping on a small change in J, T4, alpha = 0.1]")
for eps in [1e-2, 1e-3, 1e-4, 1e-6]:
    w, b, prev = 0.0, 0.0, cost(T4x, T4y, 0.0, 0.0)
    for k in range(1, 100000):
        gw, gb = grads(T4x, T4y, w, b)
        w, b = w - 0.1 * gw, b - 0.1 * gb
        J = cost(T4x, T4y, w, b)
        if abs(prev - J) < eps:
            print(f"    eps = {eps:<8g} stops at iteration {k:>5}"
                  f"  with J = {J:.6f}  (w = {w:.4f}, b = {b:.4f});"
                  f"  true minimum J = 0.025000")
            break
        prev = J

# ------------------------------------------------- 3.7 convexity
print("\n[3.7 the Hessian of J on T4 is positive definite]")
H = np.array([[(T4x ** 2).sum(), T4x.sum()], [T4x.sum(), 4.0]]) / 4
lam = np.linalg.eigvalsh(H)
print(f"    H = (1/m) [[sum x^2, sum x], [sum x, m]] = [[{H[0,0]}, {H[0,1]}],"
      f" [{H[1,0]}, {H[1,1]}]]")
print(f"    trace = {H[0,0]+H[1,1]:.4f} > 0,  det = {np.linalg.det(H):.4f} > 0,"
      f"  eigenvalues {lam[0]:.4f}, {lam[1]:.4f} > 0")
