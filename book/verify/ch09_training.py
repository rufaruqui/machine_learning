"""Chapter 9 --- numbers for training neural networks.

Network N231 is the 2-3-1 network of Chapter 8 (same weights); N231-ReLU is
the same network with ReLU in the hidden layer and a sigmoid output.

Run:  /tmp/mlenv/bin/python verify/ch09_training.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)

sig = lambda z: 1.0 / (1.0 + np.exp(-z))
relu = lambda z: np.maximum(0.0, z)

W1 = np.array([[1.0, -3.0, 5.0], [2.0, 4.0, -6.0]])
b1 = np.array([-1.0, 1.0, 2.0])
W2 = np.array([[2.0], [-1.0], [-3.0]])
b2 = np.array([0.5])
X = np.array([[1.0, 2.0], [-1.0, -1.0]])
y = np.array([1.0, 0.0])

print("[9.1] binary cross-entropy for N231 (sigmoid) outputs")
f = np.array([0.8094, 0.0562])
print(f"  -ln f     = {-np.log(f)}   (losses if y = 1)")
print(f"  -ln(1-f)  = {-np.log(1 - f)}  (losses if y = 0)")

print("\n[9.2] activation slopes")
s = lambda z: sig(z) * (1 - sig(z))
print(f"  sigmoid'(0) = {s(0):.4f}, sigmoid'(2) = {s(2):.4f}, sigmoid'(5) = {s(5):.4f}")

print("\n[9.2] N231-ReLU")
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = sig(Z2)
print("  Z1 =", Z1.tolist(), " A1 =", A1.tolist(), " Z2 =", Z2.ravel(), " A2 =", A2.ravel().round(4))
L = -(y * np.log(A2.ravel()) + (1 - y) * np.log(1 - A2.ravel()))
print(f"  per-example BCE = {L.round(4)}, cost = {L.mean():.6f}")
for x in ([0.0, 0.0], [2.0, 0.0]):
    x = np.array(x)
    z1 = x @ W1 + b1
    z2 = relu(z1) @ W2 + b2
    print(f"  x = {x}: z1 = {z1}, a1 = {relu(z1)}, z2 = {z2[0]}, a2 = {sig(z2[0]):.4g}")

print("\n[9.2] hinge sum r(x) - 2r(x-1) + 2r(x-2.5)")
h = lambda x: relu(x) - 2 * relu(x - 1) + 2 * relu(x - 2.5)
print("  values at 0, 1, 2.5, 4:", [float(h(v)) for v in (0, 1, 2.5, 4)])

print("\n[9.3] softmax")
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()
z = np.array([2.0, 1.0, 0.1])
a = softmax(z)
print(f"  e^z = {np.exp(z).round(4)}, sum = {np.exp(z).sum():.4f}, a = {a.round(4)}")
print(f"  -ln a = {(-np.log(a)).round(4)}")
print(f"  N=2: softmax([2, 0.5])_1 = {softmax(np.array([2.0, 0.5]))[0]:.4f}, sigmoid(1.5) = {sig(1.5):.4f}")
z3 = np.array([1.0, 0.0, -1.0])
print(f"  exercise: softmax([1,0,-1]) = {softmax(z3).round(4)}, -ln = {(-np.log(softmax(z3))).round(4)}")
for z1v in (3.0, 4.0):
    print(f"  coupling: softmax([{z1v}, 1, 0.1]) = {softmax(np.array([z1v, 1.0, 0.1])).round(4)}")

print("\n[9.3] gradient of softmax + cross-entropy is a - onehot(y)")
for yy in range(3):
    eps = 1e-6
    Lf = lambda zz: -np.log(softmax(zz)[yy])
    num = np.array([(Lf(z + eps * np.eye(3)[k]) - Lf(z - eps * np.eye(3)[k])) / (2 * eps) for k in range(3)])
    print(f"  y = {yy}: numerical {num.round(4)}, a - e_y {(a - np.eye(3)[yy]).round(4)}")

print("\n[9.4] round-off and overflow")
x1 = 2.0 / 10000
x2 = 1 + 1 / 10000 - (1 - 1 / 10000)
x2_32 = np.float32(1) + np.float32(1e-4) - (np.float32(1) - np.float32(1e-4))
print(f"  x1 = {x1:.20f}\n  x2 = {x2:.20f} (rel err {abs(x2 - x1) / x1:.1e})"
      f"\n  float32: {x2_32:.20f} (rel err {abs(x2_32 - x1) / x1:.1e})")
zb = np.array([1000.0, 999.0, 998.0])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(zb) / np.exp(zb).sum()
print(f"  naive softmax of [1000, 999, 998]: {naive}")
print(f"  stable: {softmax(zb).round(4)}, loss(class 0) = {np.log(1 + np.exp(-1) + np.exp(-2)):.4f}")
zl = np.array([0.0, 50.0])
exact = -(zl[0] - (zl.max() + np.log(np.exp(zl - zl.max()).sum())))
p32 = softmax(zl.astype(np.float32))
print(f"  logits [0, 50], true class 0: exact loss = {exact:.4f}; p0 = {p32[0]:.3e};"
      f" clipped at 1e-7: {-np.log(1e-7):.4f}")

print("\n[9.6] computation graph J = (wx + b - y)^2 / 2 at w=2, b=8, x=-2, y=2")
w_, b_, x_, y_ = 2.0, 8.0, -2.0, 2.0
c = w_ * x_; aa = c + b_; d = aa - y_; J = 0.5 * d * d
print(f"  c = {c}, a = {aa}, d = {d}, J = {J};  dJ/dw = {d * x_}, dJ/db = {d}")
print(f"  nudge w by 0.001: dJ = {0.5 * ((w_ + 0.001) * x_ + b_ - y_) ** 2 - J:.6f}")
w_, b_, x_, y_ = 1.0, 2.0, 3.0, 1.0
c = w_ * x_; aa = c + b_; d = aa - y_
print(f"  exercise (w=1,b=2,x=3,y=1): c={c}, a={aa}, d={d}, J={0.5 * d * d}, dJ/dw={d * x_}, dJ/db={d}")

print("\n[9.6] two ReLU layers, x = 1, y = 5")
def two_relu(p, x=1.0, yv=5.0):
    w1, bb1, w2, bb2 = p
    z1 = w1 * x + bb1; a1 = max(0.0, z1)
    z2 = w2 * a1 + bb2; a2 = max(0.0, z2)
    J = 0.5 * (a2 - yv) ** 2
    dz2 = (a2 - yv) * (z2 > 0); da1 = dz2 * w2; dz1 = da1 * (z1 > 0)
    return J, np.array([dz1 * x, dz1, dz2 * a1, dz2]), (z1, a1, z2, a2)
J, gr, fw = two_relu(np.array([2.0, 0.0, 3.0, 1.0]))
print(f"  forward {fw}, J = {J}, grads (w1, b1, w2, b2) = {gr}")
one = [(two_relu(np.array([2.0, 0, 3, 1]) + 0.001 * np.eye(4)[k])[0] - J) / 0.001 for k in range(4)]
print(f"  one-sided nudges of 0.001: {np.round(one, 4)}")
J, gr, fw = two_relu(np.array([2.0, 0.0, -1.0, 1.0]))
print(f"  exercise w2 = -1: forward {fw}, J = {J}, grads = {gr}")

print("\n[9.6] full forward + backward pass, one example x = (1, 2), y = 1, N231-ReLU")
x = np.array([1.0, 2.0]); yv = 1.0
z1 = x @ W1 + b1; a1 = relu(z1); z2 = a1 @ W2 + b2; a2 = sig(z2)
Lx = -np.log(a2[0])
dz2 = a2 - yv
dW2 = np.outer(a1, dz2); db2 = dz2
da1 = W2 @ dz2
dz1 = da1 * (z1 > 0)
dW1 = np.outer(x, dz1); db1 = dz1
print(f"  forward: z1 = {z1}, a1 = {a1}, z2 = {z2[0]:.4f}, a2 = {a2[0]:.4f}, L = {Lx:.4f}")
print(f"  dz2 = {dz2[0]:.4f}")
print(f"  dW2 = {dW2.ravel().round(4)}, db2 = {db2.round(4)}")
print(f"  da1 = {da1.round(4)}, dz1 = {dz1.round(4)}")
print(f"  dW1 =\n{dW1.round(4)}\n  db1 = {db1.round(4)}")
alpha = 1.0
W1n, b1n, W2n, b2n = W1 - alpha * dW1, b1 - alpha * db1, W2 - alpha * dW2, b2 - alpha * db2
a2n = sig(relu(x @ W1n + b1n) @ W2n + b2n)[0]
print(f"  after one step alpha = {alpha}: a2 = {a2n:.4f}, L = {-np.log(a2n):.4f}")
# numerical check of every parameter
def Lof(W1_, b1_, W2_, b2_):
    return -np.log(sig(relu(x @ W1_ + b1_) @ W2_ + b2_)[0])
eps = 1e-6; worst = 0
for P, G in [(W1, dW1), (b1, db1), (W2, dW2), (b2, db2)]:
    for idx in np.ndindex(P.shape):
        Pp = P.copy(); Pp[idx] += eps; Pm = P.copy(); Pm[idx] -= eps
        args_p = [Pp if Q is P else Q for Q in (W1, b1, W2, b2)]
        args_m = [Pm if Q is P else Q for Q in (W1, b1, W2, b2)]
        num = (Lof(*args_p) - Lof(*args_m)) / (2 * eps)
        worst = max(worst, abs(num - G[idx]))
print(f"  largest |back-prop - central difference| over 13 parameters: {worst:.1e}")

print("\n[9.6] the ReLU kink: batch gradient check on the two-example batch")
def batch_cost(P):
    W1_, b1_, W2_, b2_ = P
    A = sig(relu(X @ W1_ + b1_) @ W2_ + b2_).ravel()
    return float(np.mean(-y * np.log(A) - (1 - y) * np.log(1 - A)))
def batch_grad(P):
    W1_, b1_, W2_, b2_ = P
    Z1_ = X @ W1_ + b1_; A1_ = relu(Z1_); A2_ = sig(A1_ @ W2_ + b2_)
    dZ2 = (A2_ - y.reshape(-1, 1)) / len(y)
    dW2_ = A1_.T @ dZ2; db2_ = dZ2.sum(0)
    dZ1 = (dZ2 @ W2_.T) * (Z1_ > 0)
    return [X.T @ dZ1, dZ1.sum(0), dW2_, db2_]
def rel_check(Xb, yb, P):
    global X, y
    Xs, ys = X, y
    X, y = Xb, yb
    G = batch_grad(P); worst = 0
    for k in range(4):
        for idx in np.ndindex(P[k].shape):
            Pp = [q.copy() for q in P]; Pm = [q.copy() for q in P]
            Pp[k][idx] += 1e-6; Pm[k][idx] -= 1e-6
            num = (batch_cost(Pp) - batch_cost(Pm)) / 2e-6
            worst = max(worst, abs(num - G[k][idx]) / max(1e-8, abs(num) + abs(G[k][idx])))
    X, y = Xs, ys
    return worst
P0 = [W1, b1, W2, b2]
print(f"  lecture batch: Z1[1] = {(X @ W1 + b1)[1]}, relative difference {rel_check(X, y, P0):.1e}")
rng = np.random.default_rng(0)
Xr = rng.normal(size=(20, 2)); yr = rng.integers(0, 2, 20).astype(float)
print(f"  20 random inputs: relative difference {rel_check(Xr, yr, P0):.1e}")

print("\n[9.6] cost of computing all gradients")
for N, P in [(1e4, 1e5)]:
    print(f"  N = {N:.0e}, P = {P:.0e}: finite differences ~ N*P = {N * P:.1e}; back-prop ~ N+P = {N + P:.1e}")

print("\n[9.7] gradient descent vs Adam on J = w1^2 + 10 w2^2 from (-4, 1)")
Jq = lambda w: w[0] ** 2 + 10 * w[1] ** 2
gq = lambda w: np.array([2 * w[0], 20 * w[1]])
def run(step, alpha, n=300):
    w = np.array([-4.0, 1.0]); m = np.zeros(2); v = np.zeros(2)
    for t in range(1, n + 1):
        if Jq(w) < 0.01:
            return t - 1
        d = gq(w)
        if step == "gd":
            w = w - alpha * d
        else:
            m = 0.9 * m + 0.1 * d; v = 0.999 * v + 0.001 * d * d
            w = w - alpha * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-7)
        if not np.all(np.isfinite(w)) or np.abs(w).max() > 1e6:
            return "diverges"
    return "> 300"
for alpha in [0.003, 0.01, 0.03, 0.09, 0.1, 0.11, 0.3, 1.0, 3.0]:
    with np.errstate(over="ignore", invalid="ignore"):
        print(f"  alpha = {alpha:>5}:  GD {run('gd', alpha)!s:>9}   Adam {run('adam', alpha)!s:>6}")
w = np.array([-4.0, 1.0]); d = gq(w)
m = 0.1 * d; v = 0.001 * d * d
print(f"  first Adam step, alpha 0.3: {w - 0.3 * (m / 0.1) / (np.sqrt(v / 0.001) + 1e-7)}")

print("\n[9.8] convolution parameter counts (100 samples, 9 windows of 20, stride 10)")
print(f"  dense 100->9: {100 * 9 + 9};  separate weights per window: {9 * (20 + 1)};  one shared filter: {20 + 1}")
print(f"  windows start at: {list(range(1, 100, 10))[:9]}")
print(f"  exercise: 200 samples, window 40, stride 20 -> units = {(200 - 40) // 20 + 1}, shared params = {41}")
