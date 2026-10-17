"""Chapter 8 --- numbers for network representation and forward propagation.

Networks:
  N231  the 2-3-1 sigmoid network used throughout the chapter (and in the
        companion lecture), with hand-chosen weights
  XOR   a 2-2-1 sigmoid network whose weights are chosen so that it computes
        exclusive-or: hidden unit 1 ~ OR, hidden unit 2 ~ NAND, output ~ AND

Run:  /tmp/mlenv/bin/python verify/ch08_forward.py
"""

import numpy as np

np.set_printoptions(precision=6, suppress=True)


def g(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------- N231
W1 = np.array([[1.0, -3.0, 5.0],
               [2.0, 4.0, -6.0]])          # (n_in, units): column j is unit j
b1 = np.array([-1.0, 1.0, 2.0])
W2 = np.array([[2.0], [-1.0], [-3.0]])
b2 = np.array([0.5])


def dense_loop(a_in, W, b):
    out = np.zeros(W.shape[1])
    for j in range(W.shape[1]):
        out[j] = g(W[:, j] @ a_in + b[j])
    return out


def dense(A_in, W, b, act=g):
    return act(A_in @ W + b)


print("[8.4] N231 unit by unit")
for x in ([1.0, 2.0], [-1.0, -1.0], [0.0, 0.0]):
    x = np.array(x)
    z1 = x @ W1 + b1
    a1 = g(z1)
    z2 = a1 @ W2 + b2
    a2 = g(z2)
    print(f"  x = {x}:  z1 = {z1}  a1 = {np.round(a1, 4)}  z2 = {z2[0]:.4f}  a2 = {a2[0]:.4f}")
    assert np.allclose(dense_loop(dense_loop(x, W1, b1), W2, b2), a2)

print("\n[8.4] N231 vectorised over a batch")
X = np.array([[1.0, 2.0], [-1.0, -1.0]])
Z1 = X @ W1 + b1
A1 = g(Z1)
A2 = g(A1 @ W2 + b2)
print("  X W1      =\n", X @ W1)
print("  Z1        =\n", Z1)
print("  A1        =\n", np.round(A1, 4))
print("  A1 W2 + b2=", (A1 @ W2 + b2).ravel().round(4), "  A2 =", A2.ravel().round(4))

print("\n[8.4] parameter counts n_in*n_out + n_out")
for sizes in ([4, 3, 1], [2, 3, 1], [64, 25, 15, 1], [784, 128, 64, 10], [64, 32, 16, 1]):
    per = [a * b + b for a, b in zip(sizes, sizes[1:])]
    print(f"  {sizes}: per layer {per}, total {sum(per)}")

print("\n[8.5] XOR network")
Wx = np.array([[20.0, -20.0],
               [20.0, -20.0]])
bx = np.array([-10.0, 30.0])
wo = np.array([[20.0], [20.0]])
bo = np.array([-30.0])
for x in ([0, 0], [0, 1], [1, 0], [1, 1]):
    x = np.array(x, float)
    z = x @ Wx + bx
    a = g(z)
    zo = a @ wo + bo
    ao = g(zo)
    print(f"  x = {x.astype(int)}:  z1 = {z}  a1 = ({a[0]:.6f}, {a[1]:.6f})"
          f"  z2 = {zo[0]:.4f}  a2 = {ao[0]:.6f}  -> {int(ao[0] >= 0.5)}")
print(f"  g(10) = {g(10):.6f}, g(-10) = {g(-10):.6f}, g(30) = {g(30):.3e}, g(-30) = {g(-30):.3e}")

print("\n[8.5] no single logistic unit computes XOR: best logistic fit to the 4 points")
Xx = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
yx = np.array([0, 1, 1, 0], float)
th = np.zeros(3)
A = np.c_[np.ones(4), Xx]
for _ in range(20000):
    th -= 0.5 * A.T @ (g(A @ th) - yx) / 4
print(f"  fitted (b, w1, w2) = {th},  outputs = {g(A @ th)}")

print("\n[8.6] linear activations collapse")
Wp = W1 @ W2
bp = b1 @ W2 + b2
print(f"  W' = {Wp.ravel()},  b' = {bp}")
for x in ([1.0, 2.0], [-1.0, -1.0]):
    x = np.array(x)
    two = (x @ W1 + b1) @ W2 + b2
    one = x @ Wp + bp
    print(f"  x = {x}: two layers {two[0]:.4f}, one layer {one[0]:.4f}")

print("\n[8.7] matrix multiplication example")
Am = np.array([[1, 2], [3, 4]])
Bm = np.array([[5, 0, 1], [-1, 2, 0]])
print("  AB =\n", Am @ Bm)

print("\n[8.8] shapes")
x1d = np.array([200.0, 17.0])
print(f"  1-D {x1d.shape}, row {x1d.reshape(1, -1).shape}, column {x1d.reshape(-1, 1).shape}")
try:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from tensorflow import keras
    net = keras.Sequential([keras.Input(shape=(2,)),
                            keras.layers.Dense(3, activation="sigmoid"),
                            keras.layers.Dense(1, activation="sigmoid")])
    net.layers[0].set_weights([W1, b1])
    net.layers[1].set_weights([W2, b2])
    out = net.predict(X, verbose=0).ravel()
    print(f"  Keras with the same weights: {np.round(out, 4)}"
          f"  (max diff from NumPy {np.abs(out - A2.ravel()).max():.1e})")
except Exception as exc:  # TensorFlow is optional for this script
    print(f"  (Keras check skipped: {type(exc).__name__})")

print("\n[exercises] answers")
x = np.array([0.5, -0.5])
z1 = x @ W1 + b1
print(f"  N231 at x = (0.5, -0.5): z1 = {z1}, a2 = {g(g(z1) @ W2 + b2)[0]:.4f}")
Xe = np.array([[0.0, 0.0], [2.0, 1.0]])
print("  N231 batch [[0,0],[2,1]]: Z1 =\n", Xe @ W1 + b1, "\n  A2 =", g(g(Xe @ W1 + b1) @ W2 + b2).ravel().round(4))
