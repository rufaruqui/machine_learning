"""Chapter 2 --- numbers for the cost function chapter.

Two datasets:
  T3  the three-point set (1,1), (2,2), (3,3)   -- used for the J(w) parabola
  T4  the four-point set  (1,1), (2,3), (3,4), (4,6) -- used for the bowl,
      the contour map, and the fit done by hand
plus D1, the twelve apartments of Chapter 1.

Run:  /tmp/mlenv/bin/python verify/ch02_cost_function.py
"""

import numpy as np

# ---------------------------------------------------------------- datasets
T3x = np.array([1.0, 2.0, 3.0]);            T3y = np.array([1.0, 2.0, 3.0])
T4x = np.array([1.0, 2.0, 3.0, 4.0]);       T4y = np.array([1.0, 3.0, 4.0, 6.0])
D1x = np.array([5, 7, 9, 10, 12, 13, 15, 16, 18, 20, 22, 24], dtype=float)
D1y = np.array([22, 26, 36, 33, 44, 49, 52, 61, 63, 72, 74, 84], dtype=float)


def cost(x, y, w, b):
    return float(((w * x + b - y) ** 2).sum() / (2 * x.size))


def fit(x, y):
    w = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    return float(w), float(y.mean() - w * x.mean())


# ------------------------------------------------- 2.3  why 1/m, why one half
print("[2.3 why the 1/m]  T4, and T4 with every example duplicated")
T4x2, T4y2 = np.concatenate([T4x, T4x]), np.concatenate([T4y, T4y])
for w, b in [(1.6, -0.5), (1.0, 1.0)]:
    sse = float(((w * T4x + b - T4y) ** 2).sum())
    sse2 = float(((w * T4x2 + b - T4y2) ** 2).sum())
    print(f"    (w,b)=({w},{b}):  sum sq  {sse:7.3f} -> {sse2:7.3f} (doubles);"
          f"   J  {cost(T4x,T4y,w,b):.4f} -> {cost(T4x2,T4y2,w,b):.4f} (same)")

# ------------------------------------------------- 2.4  the parabola, b fixed
print("\n[2.4 J(w) on T3 with b = 0]   J(w) = (14/6)(w-1)^2")
for w in [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
    print(f"    w = {w:4.1f}   J = {cost(T3x, T3y, w, 0.0):.4f}"
          f"   check (14/6)(w-1)^2 = {14 / 6 * (w - 1) ** 2:.4f}")

# ------------------------------------------------- 2.5  the bowl and contours
Sx, Sxx, Sy, Sxy, Syy = (T4x.sum(), (T4x ** 2).sum(), T4y.sum(),
                         (T4x * T4y).sum(), (T4y ** 2).sum())
m = T4x.size
print(f"\n[2.5 T4 sums]  m={m}  Sx={Sx:.0f}  Sxx={Sxx:.0f}  Sy={Sy:.0f}"
      f"  Sxy={Sxy:.0f}  Syy={Syy:.0f}")
print(f"    J(w,b) = ({Sxx:.0f}w^2 + {2*Sx:.0f}wb + {m}b^2"
      f" - {2*Sxy:.0f}w - {2*Sy:.0f}b + {Syy:.0f}) / {2*m}")
wstar, bstar = fit(T4x, T4y)
print(f"    minimum at w = {wstar:.4f}, b = {bstar:.4f},"
      f" J = {cost(T4x, T4y, wstar, bstar):.4f}")

print("\n[2.5 three parameter pairs marked on the contour map]")
for w, b in [(wstar, bstar), (1.0, 1.0), (2.2, -2.0), (1.6, 0.5), (0.8, 0.0)]:
    print(f"    (w,b) = ({w:6.4f}, {b:7.4f})   J = {cost(T4x, T4y, w, b):8.4f}")

# Hessian: how elongated are those contours?
H = np.array([[Sxx, Sx], [Sx, float(m)]]) / m
lam = np.linalg.eigvalsh(H)
print(f"    Hessian eigenvalues {lam[0]:.4f}, {lam[1]:.4f}"
      f"  -> axis ratio {np.sqrt(lam.max()/lam.min()):.2f}")

# ------------------------------------------------- 2.6  fitting by hand
print("\n[2.6 closed form on T4]")
xb, yb = T4x.mean(), T4y.mean()
print(f"    xbar = {xb:.4f}   ybar = {yb:.4f}")
print(f"    Sxy - Sx*Sy/m = {Sxy - Sx*Sy/m:.4f}")
print(f"    Sxx - Sx^2/m  = {Sxx - Sx**2/m:.4f}")
print(f"    w = {wstar:.4f}   b = {bstar:.4f}")
print("    residuals at the optimum:")
r = wstar * T4x + bstar - T4y
for xi, yi, ri in zip(T4x, T4y, r):
    print(f"      x={xi:.0f}  y={yi:.0f}  yhat={wstar*xi+bstar:6.3f}  r={ri:+6.3f}")
print(f"    sum of residuals      = {r.sum():+.12f}   (must be 0)")
print(f"    sum of residual * x   = {(r*T4x).sum():+.12f}   (must be 0)")
print(f"    J = sum r^2 / 2m      = {float((r**2).sum()):.4f} / {2*m} ="
      f" {cost(T4x,T4y,wstar,bstar):.4f}")

print("\n[2.6 the same closed form reproduces Chapter 1's apartment fit]")
w1, b1 = fit(D1x, D1y)
print(f"    w = {w1:.4f}   b = {b1:.4f}   J = {cost(D1x,D1y,w1,b1):.4f}")

# ------------------------------------------------- 2.8  a low cost is not a good model
print("\n[2.8 one outlier moves the fit]  T4 plus the point (5, 30)")
Ox, Oy = np.append(T4x, 5.0), np.append(T4y, 30.0)
wo, bo = fit(Ox, Oy)
print(f"    before: w = {wstar:.4f}  b = {bstar:.4f}")
print(f"    after : w = {wo:.4f}  b = {bo:.4f}")
for xq, yq in [(1.0, 1.0), (4.0, 6.0)]:
    print(f"    prediction at x = {xq:.0f}: {wstar*xq+bstar:7.3f}  ->"
          f" {wo*xq+bo:7.3f}   (observed y = {yq:.0f})")
print(f"    J on the five points  = {cost(Ox,Oy,wo,bo):.4f}")
print(f"    J of the old line there = {cost(Ox,Oy,wstar,bstar):.4f}")
