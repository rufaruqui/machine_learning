"""Chapter 11 --- numbers for error analysis, skewed classes and fairness.

Screening model S: a score s with s ~ N(3, 1) for positives and N(0, 1) for
negatives, prevalence pi = 0.005, and the classifier reports the exact
posterior P(y = 1 | s).  All rates are computed analytically from the normal
CDF, so they are exact for this model.

Fairness example: the same model applied to two groups of equal prevalence
(5%) in which the score separates the classes less well for group B
(positives ~ N(2.2, 1)) than for group A (positives ~ N(3, 1)).

Run:  /tmp/mlenv/bin/python verify/ch11_skew.py
"""

from math import erf, exp, log, sqrt

import numpy as np

Phi = lambda t: 0.5 * (1 + erf(t / sqrt(2)))


def post_to_score(p, mu1, pi):
    """Score s at which P(y=1|s) = p, for N(mu1,1) vs N(0,1) and prior pi."""
    # log-odds(s) = mu1*s - mu1^2/2 + log(pi/(1-pi))
    return (log(p / (1 - p)) - log(pi / (1 - pi)) + mu1 ** 2 / 2) / mu1


def rates(t, mu1):
    """Recall (TPR) and false-positive rate for threshold t on the score."""
    return 1 - Phi(t - mu1), 1 - Phi(t)


print("[11.2] error analysis arithmetic")
print(f"  misspellings: 3 of 100 sampled -> about {3 / 100 * 500:.0f} of 500 errors")
print(f"  pharmaceutical + phishing: {21 + 18} of 100")

print("\n[11.4] transfer learning parameter counts (784-128-64-1000 -> 10 classes)")
p1, p2, head_old, head_new = 784 * 128 + 128, 128 * 64 + 64, 64 * 1000 + 1000, 64 * 10 + 10
print(f"  layer 1 {p1}, layer 2 {p2}, old head {head_old}, new head {head_new}")
print(f"  copied {p1 + p2}; option 1 trains {head_new} of {p1 + p2 + head_new}"
      f" ({100 * head_new / (p1 + p2 + head_new):.2f}%)")

print("\n[11.5] the accuracy trap at 0.5% prevalence (10,000 patients)")
TP, FN, FP, TN = 35, 15, 90, 9860
m = TP + FN + FP + TN
P = TP / (TP + FP); R = TP / (TP + FN); F1 = 2 * P * R / (P + R)
print(f"  positives {TP + FN}, accuracy {(TP + TN) / m:.4f}, always-negative accuracy {(FP + TN) / m:.4f}")
print(f"  precision {P:.4f}, recall {R:.4f}, F1 {F1:.4f}, specificity {TN / (TN + FP):.4f}")

print("\n[11.6] F1 versus the arithmetic mean")
for p, r in [(0.5, 0.4), (0.7, 0.1), (0.02, 1.0), (0.9, 0.2)]:
    print(f"  P = {p}, R = {r}: mean {(p + r) / 2:.3f}, F1 {2 * p * r / (p + r):.4f}")
print(f"  exercise: TP 40, FP 60, FN 20, TN 9880 -> P {40 / 100:.3f}, R {40 / 60:.4f},"
      f" F1 {2 * 0.4 * (2 / 3) / (0.4 + 2 / 3):.4f}, accuracy {(40 + 9880) / 10000:.4f}")

MU1, PI, N = 3.0, 0.005, 10000
print(f"\n[11.6] screening model S (mu1 = {MU1}, pi = {PI}): threshold sweep on P(y=1|x)")
best = (0, None)
for p in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    t = post_to_score(p, MU1, PI)
    R_, F_ = rates(t, MU1)
    tp, fp = PI * N * R_, (1 - PI) * N * F_
    Pp = tp / (tp + fp)
    F1_ = 2 * Pp * R_ / (Pp + R_)
    print(f"  t = {p:4}: score {t:6.3f}  recall {R_:.3f}  precision {Pp:.3f}  F1 {F1_:.3f}"
          f"  per 10,000: TP {tp:5.1f}  FP {fp:6.1f}  FN {PI * N - tp:5.1f}")
grid = np.linspace(0.001, 0.999, 9981)
f1s = []
for p in grid:
    t = post_to_score(p, MU1, PI)
    R_, F_ = rates(t, MU1)
    tp, fp = PI * R_, (1 - PI) * F_
    Pp = tp / (tp + fp)
    f1s.append(2 * Pp * R_ / (Pp + R_))
i = int(np.argmax(f1s))
print(f"  F1 is maximised at t = {grid[i]:.3f} with F1 = {f1s[i]:.3f}")

print("\n[11.6] precision-recall curve points (score thresholds -1..6)")
pts = []
for t in np.arange(-1.0, 6.01, 0.25):
    R_, F_ = rates(t, MU1)
    tp, fp = PI * R_, (1 - PI) * F_
    pts.append(f"({R_:.3f},{tp / (tp + fp):.3f})")
print("  " + "".join(pts))

print("\n[11.7] threshold from costs: a missed case costs 50 false alarms")
cFN, cFP = 50.0, 1.0
print(f"  Bayes threshold c_FP/(c_FP + c_FN) = {cFP / (cFP + cFN):.4f}")
def cost_at(p):
    t = post_to_score(p, MU1, PI)
    R_, F_ = rates(t, MU1)
    fn, fp = PI * N * (1 - R_), (1 - PI) * N * F_
    return cFN * fn + cFP * fp, fn, fp
for p in [0.005, 0.0196, 0.05, 0.1, grid[i], 0.5]:
    c, fn, fp = cost_at(p)
    print(f"  t = {p:.4f}: FN {fn:5.2f}  FP {fp:7.1f}  cost per 10,000 = {c:7.1f}")
cs = [cost_at(p)[0] for p in grid]
j = int(np.argmin(cs))
print(f"  cost minimised at t = {grid[j]:.4f}, cost {cs[j]:.1f}")

print("\n[11.8] subgroup evaluation: one threshold, two groups (prevalence 5% each)")
PIg = 0.05
t_score = post_to_score(0.5, 3.0, PIg)      # threshold tuned on group A's model
for name, mu in [("A", 3.0), ("B", 2.2)]:
    R_, F_ = rates(t_score, mu)
    acc = PIg * R_ + (1 - PIg) * (1 - F_)
    tp, fp = PIg * R_, (1 - PIg) * F_
    print(f"  group {name}: recall {R_:.3f}  FNR {1 - R_:.3f}  FPR {F_:.4f}"
          f"  precision {tp / (tp + fp):.3f}  accuracy {acc:.4f}")
print(f"  common score threshold: {t_score:.3f}")
# a threshold that equalises recall
for name, mu in [("B", 2.2)]:
    tB = mu + (t_score - 3.0)        # same distance from the positive mean
    R_, F_ = rates(tB, mu)
    print(f"  group B threshold for equal recall: score {tB:.3f} -> recall {R_:.3f}, FPR {F_:.4f}")
# pooled accuracy with 90% A, 10% B
RA, FA = rates(t_score, 3.0); RB, FB = rates(t_score, 2.2)
accA = PIg * RA + (1 - PIg) * (1 - FA); accB = PIg * RB + (1 - PIg) * (1 - FB)
print(f"  pooled accuracy (90% A, 10% B): {0.9 * accA + 0.1 * accB:.4f}")
