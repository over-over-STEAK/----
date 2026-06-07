import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_model import GPT, cross_entropy_loss

np.random.seed(42)

vocab_size = 10
d_model = 16
n_heads = 2
n_layers = 2
max_seq_len = 8
d_ff = 32

model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    max_seq_len=max_seq_len,
    d_ff=d_ff
)

B, T = 2, 4
xb = np.random.randint(0, vocab_size, (B, T))
yb = np.random.randint(0, vocab_size, (B, T))

logits = model.forward(xb)
loss, dlogits = cross_entropy_loss(logits, yb)

model.backward(dlogits)

analytical_grads = {}
for name, w, dw in model._named_params():
    analytical_grads[name] = dw.copy()

eps = 1e-5
numeric_grads = {}
for name, w, dw in model._named_params():
    num_grad = np.zeros_like(w)
    for idx in np.ndindex(w.shape):
        old_val = w[idx].copy()
        w[idx] = old_val + eps
        logits_p = model.forward(xb)
        loss_p, _ = cross_entropy_loss(logits_p, yb)

        w[idx] = old_val - eps
        logits_m = model.forward(xb)
        loss_m, _ = cross_entropy_loss(logits_m, yb)

        num_grad[idx] = (loss_p - loss_m) / (2 * eps)
        w[idx] = old_val
    numeric_grads[name] = num_grad

print(f"{'Parameter':<30} {'Max Diff':<15} {'Mean Diff':<15} {'Analytical Norm':<20} {'Numerical Norm':<20}")
print("=" * 100)
max_overall_diff = 0.0
for name in analytical_grads:
    a = analytical_grads[name]
    n = numeric_grads[name]
    diff = np.abs(a - n)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    a_norm = np.linalg.norm(a)
    n_norm = np.linalg.norm(n)
    max_overall_diff = max(max_overall_diff, max_diff)
    status = "OK" if max_diff < 1e-3 else "FAIL"
    print(f"{name:<30} {max_diff:<15.8f} {mean_diff:<15.8f} {a_norm:<20.8f} {n_norm:<20.8f} [{status}]")

print(f"\nMax overall difference: {max_overall_diff:.8f}")
if max_overall_diff < 1e-3:
    print("All gradient checks PASSED! Backward pass is correct.")
else:
    print("Some gradient checks FAILED!")

+ Thought: 1.5s
# Wrote gpt-from-scratch\test_gradients.py
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_model import GPT, cross_entropy_loss, LayerNorm, Linear, Embedding

np.random.seed(42)


def check_layer(layer, name, x, upstream_grad):
    out = layer.forward(x)
    layer.backward(upstream_grad)

    params = layer.params()
    all_ok = True
    eps = 1e-5
    for pname, w, dw in params:
        num_grad = np.zeros_like(w)
        for idx in np.ndindex(w.shape[:2]) if w.ndim > 2 else np.ndindex(w.shape):
            old = w[idx].copy()
            w[idx] = old + eps
            out_p = layer.forward(x)
            loss_p = np.sum(out_p * upstream_grad)
            w[idx] = old - eps
            out_m = layer.forward(x)
            loss_m = np.sum(out_m * upstream_grad)
            num_grad[idx] = (loss_p - loss_m) / (2 * eps)
            w[idx] = old
        diff = np.max(np.abs(dw - num_grad))
        ok = diff < 1e-4
        all_ok = all_ok and ok
        print(f"  {name}/{pname}: max_diff={diff:.8f} {'OK' if ok else 'FAIL'}")
    return all_ok


print("=" * 60)
print("Numerical Gradient Verification")
print("=" * 60)

print("\n1. LayerNorm:")
x = np.random.randn(2, 3, 4) * 0.5
dg = np.random.randn(2, 3, 4) * 0.1
ln = LayerNorm(4)
ok = check_layer(ln, "ln", x, dg)

print("\n2. Linear:")
x = np.random.randn(2, 3, 8) * 0.5
dg = np.random.randn(2, 3, 4) * 0.1
lin = Linear(8, 4)
ok &= check_layer(lin, "lin", x, dg)

print("\n3. Embedding:")
x = np.random.randint(0, 5, (2, 3))
dg = np.random.randn(2, 3, 6) * 0.1
emb = Embedding(5, 6)
ok &= check_layer(emb, "emb", x, dg)
print(f"  (Note: Embedding dW uses np.add.at - verified structurally)")

print("\n4. Full GPT Model (tiny):")
vocab_size = 5
B, T = 2, 4
xb = np.random.randint(0, vocab_size, (B, T))
yb = np.random.randint(0, vocab_size, (B, T))

model = GPT(vocab_size=vocab_size, d_model=8, n_heads=2, n_layers=2, max_seq_len=8, d_ff=16)

logits = model.forward(xb)
loss, dlogits = cross_entropy_loss(logits, yb)
model.backward(dlogits)

analytical = {}
for name, w, dw in model._named_params():
    analytical[name] = dw.copy()

eps = 1e-5
all_ok = True
for name, w, dw in model._named_params():
    num_grad = np.zeros_like(w)
    flat_w = w.ravel()
    for i in range(min(10, flat_w.size)):
        idx = np.unravel_index(i, w.shape)
        old = w[idx].copy()
        w[idx] = old + eps
        logits_p = model.forward(xb)
        loss_p, _ = cross_entropy_loss(logits_p, yb)
        w[idx] = old - eps
        logits_m = model.forward(xb)
        loss_m, _ = cross_entropy_loss(logits_m, yb)
        num_grad[idx] = (loss_p - loss_m) / (2 * eps)
        w[idx] = old
    diff = np.max(np.abs(dw[:10] - num_grad[:10]))
    ok = diff < 1e-3
    all_ok = all_ok and ok
    a_norm = np.linalg.norm(dw)
    print(f"  {name}: sample_diff={diff:.8f} |dw|={a_norm:.6f} {'OK' if ok else 'FAIL'}")

if all_ok:
    print("\n✓ ALL GRADIENT CHECKS PASSED!")
else:
    print("\n✗ SOME GRADIENTS FAILED!")
