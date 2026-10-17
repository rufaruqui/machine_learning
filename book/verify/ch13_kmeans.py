"""Chapter 13 --- numbers for k-means clustering.

Datasets:
  K6   six points in the plane, worked by hand
  B60  60 points in three blobs (sklearn make_blobs, centres (2,8), (8,8),
       (5,2.5), sd 1.1, random_state 7) --- also used in the companion lecture
  U60  60 points uniform on [0,10]^2 (no cluster structure)
  IMG  sklearn's bundled sample photograph 'china.jpg' (427 x 640 pixels)

Run:  /tmp/mlenv/bin/python verify/ch13_kmeans.py
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def kmeans(X, mu, max_iter=300):
    mu = np.array(mu, float)
    hist = []
    for _ in range(max_iter):
        d = ((X[:, None] - mu[None]) ** 2).sum(2)
        c = d.argmin(1)
        if len(np.unique(c)) < len(mu):
            return None, None, hist, "empty"
        J_assign = float(np.mean(d[np.arange(len(X)), c]))
        mu2 = np.array([X[c == k].mean(0) for k in range(len(mu))])
        J_move = float(np.mean(((X - mu2[c]) ** 2).sum(1)))
        hist.append((c.copy(), mu.copy(), J_assign, mu2.copy(), J_move))
        if np.allclose(mu2, mu):
            break
        mu = mu2
    return c, mu, hist, "ok"


def J(X, c, mu):
    return float(np.mean(((X - mu[c]) ** 2).sum(1)))


print("[13.2] six points by hand, start mu1 = A, mu2 = B")
K6 = np.array([[1, 1], [1.5, 2], [5, 7], [3.5, 5], [4.5, 5], [3.5, 4.5]])
c, mu, hist, _ = kmeans(K6, [[1, 1], [1.5, 2]])
for it, (cc, m_old, Ja, m_new, Jm) in enumerate(hist, 1):
    d = ((K6[:, None] - m_old[None]) ** 2).sum(2)
    print(f"  iteration {it}: distances^2 to (mu1, mu2):")
    for name, row in zip("ABCDEF", d):
        print(f"     {name}: {row[0]:8.4f} {row[1]:8.4f}")
    print(f"   assignment {cc + 1}  J = {Ja:.4f};  new centroids {m_new.tolist()}  J = {Jm:.4f}")

from sklearn.datasets import make_blobs
B60, _ = make_blobs(n_samples=60, centers=[[2, 8], [8, 8], [5, 2.5]], cluster_std=1.1, random_state=7)

print("\n[13.4] random initialisation on B60, K = 3, 200 runs")
res = []
for s in range(200):
    r = np.random.default_rng(s)
    c, mu, hist, st = kmeans(B60, B60[r.choice(60, 3, replace=False)])
    res.append(round(J(B60, c, mu), 3) if st == "ok" else "empty")
vals, cnt = np.unique([v for v in res if v != "empty"], return_counts=True)
print("  final J values and counts:", dict(zip(vals.tolist(), cnt.tolist())), " empty:", res.count("empty"))
print("  bunched start (1,1),(2,1.5),(1.5,3):", kmeans(B60, [[1, 1], [2, 1.5], [1.5, 3]])[3])


def kmeanspp(X, K, rng):
    idx = [rng.integers(len(X))]
    for _ in range(K - 1):
        d2 = ((X[:, None] - X[idx][None]) ** 2).sum(2).min(1)
        idx.append(rng.choice(len(X), p=d2 / d2.sum()))
    return X[idx]


print("\n[13.5] k-means++ initialisation, 200 runs")
res = []
for s in range(200):
    c, mu, hist, st = kmeans(B60, kmeanspp(B60, 3, np.random.default_rng(s)))
    res.append(round(J(B60, c, mu), 3))
vals, cnt = np.unique(res, return_counts=True)
print("  final J values and counts:", dict(zip(vals.tolist(), cnt.tolist())))
r = np.random.default_rng(3)
first = r.integers(60)
d2 = ((B60 - B60[first]) ** 2).sum(1)
print(f"  example: first centre index {first}, probability of choosing the farthest point next "
      f"{d2.max() / d2.sum():.4f} vs uniform {1 / 60:.4f}")


def best_J(X, K, runs=50):
    best = np.inf
    for s in range(runs):
        r = np.random.default_rng(1000 + s)
        c, mu, h, st = kmeans(X, X[r.choice(len(X), K, replace=False)])
        if st == "ok":
            best = min(best, J(X, c, mu))
    return best


print("\n[13.6] elbow curves (best of 50 random starts)")
U60 = np.random.default_rng(5).uniform(0, 10, (60, 2))
for K in range(1, 9):
    print(f"  K = {K}:  B60 {best_J(B60, K):7.3f}   U60 {best_J(U60, K):7.3f}")

print("\n[13.7] image compression")
try:
    from sklearn.datasets import load_sample_image
    img = load_sample_image("china.jpg") / 255.0
    h, w, _ = img.shape
    P = img.reshape(-1, 3)
    rng = np.random.default_rng(0)
    sample = P[rng.choice(len(P), 5000, replace=False)]
    print(f"  image {h} x {w} = {h * w} pixels")
    for K in (2, 4, 8, 16, 32):
        best = None
        for s in range(5):
            mu0 = sample[np.random.default_rng(s).choice(5000, K, replace=False)]
            c, mu, hist, st = kmeans(sample, mu0)
            if st == "ok" and (best is None or J(sample, c, mu) < best[0]):
                best = (J(sample, c, mu), mu)
        mu = best[1]
        cfull = np.empty(len(P), int)
        for i in range(0, len(P), 50000):
            cfull[i:i + 50000] = ((P[i:i + 50000, None] - mu[None]) ** 2).sum(2).argmin(1)
        mse = float(np.mean(((P - mu[cfull]) ** 2).sum(1)))
        bits = int(np.ceil(np.log2(K)))
        orig = h * w * 24
        comp = h * w * bits + K * 24
        print(f"  K = {K:2d}: bits/pixel {bits}, size {comp / 8 / 1024:7.1f} KiB vs {orig / 8 / 1024:7.1f} KiB"
              f" (factor {orig / comp:5.2f}), mean squared colour error {mse:.5f}")
        if K == 16:
            order = np.argsort(mu.sum(1))
            print("  K=16 palette (RGB 0-255, dark to light):",
                  [tuple(int(round(255 * v)) for v in mu[k]) for k in order])
            counts = np.bincount(cfull, minlength=16)[order]
            print("  pixel share per palette colour:", np.round(counts / len(P), 4).tolist())
except Exception as exc:
    print("  skipped:", exc)

print("\n[13.3] snapshots of one run on B60 (start: three random examples, seed 15)")
r = np.random.default_rng(15)
mu = B60[r.choice(60, 3, replace=False)].astype(float)
snaps = {}
step = 0
while True:
    d = ((B60[:, None] - mu[None]) ** 2).sum(2)
    c = d.argmin(1)
    snaps[step] = (c.copy(), mu.copy(), float(np.mean(d[np.arange(60), c])))
    mu2 = np.array([B60[c == k].mean(0) for k in range(3)])
    if np.allclose(mu2, mu):
        break
    mu = mu2
    step += 1
print(f"  converged after {step + 1} assignment steps")
for s in (0, 2, 4, step):
    c, m, Jv = snaps[s]
    print(f"  snapshot iteration {s + 1}: J = {Jv:.3f}, centroids {np.round(m, 2).tolist()}")
    for k in range(3):
        pts = "".join(f"({a:.2f},{b:.2f})" for a, b in B60[c == k])
        print(f"    S{s}_{k}: {pts}")
