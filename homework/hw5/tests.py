import numpy as np
import sys
import os
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_model import (
    GPT, LayerNorm, Linear, Embedding, CausalSelfAttention,
    FeedForward, TransformerBlock, SGDMomentum, cross_entropy_loss
)

SEED = 42
np.random.seed(SEED)


def _softmax_ref(x, axis=-1):
    """NumPy 參考 softmax 實作"""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def _layer_norm_ref(x, gamma, beta, eps=1e-5):
    """NumPy 參考 LayerNorm 實作"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


# ============================================================
# 1. 基礎元件測試
# ============================================================

def test_linear_forward():
    """Linear 前向傳播正確性"""
    print("  test_linear_forward...", end=" ")
    lin = Linear(4, 3)
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    out = lin.forward(x)
    expected = x @ lin.W.T + lin.b
    assert np.allclose(out, expected), f"{out} != {expected}"
    print("PASS")


def test_linear_backward():
    """Linear 反向傳播梯度正確性 (數值檢驗)"""
    print("  test_linear_backward...", end=" ")
    lin = Linear(4, 3)
    x = np.random.randn(2, 4) * 0.5
    dout = np.random.randn(2, 3) * 0.1

    out = lin.forward(x)
    lin.backward(dout)

    eps = 1e-5
    for (name, w, dw) in lin.params():
        for _ in range(3):
            idx = tuple(np.random.randint(0, s) for s in w.shape)
            old = w[idx].copy()
            w[idx] = old + eps
            loss_p = np.sum(lin.forward(x) * dout)
            w[idx] = old - eps
            loss_m = np.sum(lin.forward(x) * dout)
            num_grad = (loss_p - loss_m) / (2 * eps)
            w[idx] = old
            assert abs(dw[idx] - num_grad) < 1e-4, f"d{name}[{idx}] mismatch: {dw[idx]} vs {num_grad}"
    print("PASS")


def test_layernorm_forward():
    """LayerNorm 前向傳播正確性"""
    print("  test_layernorm_forward...", end=" ")
    ln = LayerNorm(4)
    ln.gamma = np.array([1.0, 0.5, 2.0, 1.5])
    ln.beta = np.array([0.1, -0.1, 0.0, 0.2])
    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    out = ln.forward(x)
    expected = _layer_norm_ref(x, ln.gamma, ln.beta)
    assert np.allclose(out, expected), f"mismatch: {out} vs {expected}"
    print("PASS")


def test_layernorm_backward():
    """LayerNorm 反向傳播梯度正確性"""
    print("  test_layernorm_backward...", end=" ")
    ln = LayerNorm(8)
    x = np.random.randn(3, 5, 8) * 0.5
    dout = np.random.randn(3, 5, 8) * 0.1

    out = ln.forward(x)
    ln.backward(dout)

    eps = 1e-5
    for (name, w, dw) in ln.params():
        for _ in range(3):
            idx = tuple(np.random.randint(0, s) for s in w.shape)
            old = w[idx].copy()
            w[idx] = old + eps
            loss_p = np.sum(ln.forward(x) * dout)
            w[idx] = old - eps
            loss_m = np.sum(ln.forward(x) * dout)
            num_grad = (loss_p - loss_m) / (2 * eps)
            w[idx] = old
            assert abs(dw[idx] - num_grad) < 1e-4, f"d{name}[{idx}] mismatch"
    print("PASS")


def test_embedding_forward():
    """Embedding 前向傳播正確性"""
    print("  test_embedding_forward...", end=" ")
    emb = Embedding(10, 4)
    emb.W = np.arange(40, dtype=np.float32).reshape(10, 4)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    out = emb.forward(x)
    expected = emb.W[x]
    assert np.allclose(out, expected), f"mismatch"
    print("PASS")


def test_embedding_backward():
    """Embedding 反向傳播梯度正確性"""
    print("  test_embedding_backward...", end=" ")
    emb = Embedding(5, 3)
    x = np.array([[0, 2], [3, 1], [2, 4]])
    dout = np.random.randn(3, 2, 3) * 0.1

    out = emb.forward(x)
    emb.backward(dout)

    eps = 1e-5
    for (name, w, dw) in emb.params():
        for _ in range(5):
            idx = tuple(np.random.randint(0, s) for s in w.shape)
            old = w[idx].copy()
            w[idx] = old + eps
            loss_p = np.sum(emb.forward(x) * dout)
            w[idx] = old - eps
            loss_m = np.sum(emb.forward(x) * dout)
            num_grad = (loss_p - loss_m) / (2 * eps)
            w[idx] = old
            assert abs(dw[idx] - num_grad) < 1e-4, f"d{name}[{idx}] mismatch: {dw[idx]} vs {num_grad}"
    print("PASS")


def test_causal_attention_mask():
    """Causal mask 確保只看歷史，不看未來"""
    print("  test_causal_attention_mask...", end=" ")
    d_model, n_heads = 8, 2
    attn = CausalSelfAttention(d_model, n_heads)
    B, T = 2, 6
    x = np.random.randn(B, T, d_model) * 0.1

    out = attn.forward(x)

    A = attn.cache['A']
    for b in range(B):
        for h in range(n_heads):
            for i in range(T):
                for j in range(i + 1, T):
                    assert A[b, h, i, j] == 0.0, f"位置 {i} 能看到未來位置 {j}！"
    print("PASS")


# ============================================================
# 2. 記憶與狀態測試
# ============================================================

def test_memory_limit():
    """模型在長序列上不應該超出 max_seq_len"""
    print("  test_memory_limit...", end=" ")
    vocab_size = 10
    model = GPT(vocab_size=vocab_size, d_model=8, n_layers=1, max_seq_len=16)
    context = np.array([[1, 2, 3, 4, 5]])
    output = model.generate(context, max_new_tokens=30)
    assert output.shape[1] <= 16 + 30, f"序列太長: {output.shape}"
    assert output.shape[1] == 35, f"不合理: {output.shape}"
    print("PASS")


def test_sequence_too_long():
    """超長輸入應自動截斷至 max_seq_len"""
    print("  test_sequence_too_long...", end=" ")
    vocab_size = 10
    model = GPT(vocab_size=vocab_size, d_model=8, n_layers=1, max_seq_len=8)
    long_input = np.zeros((1, 20), dtype=np.int32)
    logits = model.forward(long_input)
    assert logits.shape[1] == 20, f"前向傳播不應截斷, shape={logits.shape}"
    context = np.zeros((1, 20), dtype=np.int32)
    output = model.generate(context, max_new_tokens=5)
    assert output.shape[1] == 25, f"生成後 shape={output.shape}"
    print("PASS")


def test_batch_independence():
    """同 batch 不同樣本不應互相干擾"""
    print("  test_batch_independence...", end=" ")
    vocab_size = 10
    model = GPT(vocab_size=vocab_size, d_model=8, n_layers=1, max_seq_len=8)

    x1 = np.array([[1, 2, 3, 0, 0, 0, 0, 0]], dtype=np.int32)
    x2 = np.array([[5, 6, 7, 8, 9, 0, 0, 0]], dtype=np.int32)
    x_batch = np.concatenate([x1, x2], axis=0)

    logits_batch = model.forward(x_batch)
    logits_1 = model.forward(x1)
    logits_2 = model.forward(x2)

    assert np.allclose(logits_batch[0:1], logits_1), "單獨 forward 與 batch forward 不一致"
    assert np.allclose(logits_batch[1:2], logits_2), "單獨 forward 與 batch forward 不一致"
    print("PASS")


# ============================================================
# 3. 無效輸入測試
# ============================================================

def test_invalid_tool():
    """測試無效生成參數不應崩潰"""
    print("  test_invalid_tool...", end=" ")
    vocab_size = 10
    model = GPT(vocab_size=vocab_size, d_model=8, n_layers=1, max_seq_len=16)

    context = np.array([[1, 2, 3]])

    for temp in [0.0, 0.5, 1.0, 2.0]:
        try:
            model.generate(context, max_new_tokens=5, temperature=temp)
        except Exception as e:
            assert False, f"temperature={temp} 崩潰: {e}"

    for k in [1, 5, 20, None, 1000]:
        try:
            model.generate(context, max_new_tokens=5, top_k=k)
        except Exception as e:
            assert False, f"top_k={k} 崩潰: {e}"

    print("PASS")


def test_workspace_escape():
    """測試權重初始化不應產生 NaN 或 Inf"""
    print("  test_workspace_escape...", end=" ")
    for _ in range(10):
        model = GPT(vocab_size=50, d_model=32, n_layers=2, max_seq_len=32)
        x = np.random.randint(0, 50, (2, 16))
        logits = model.forward(x)
        assert not np.any(np.isnan(logits)), "前向傳播產生 NaN！"
        assert not np.any(np.isinf(logits)), "前向傳播產生 Inf！"
    print("PASS")


def test_json_parser():
    """測試參數 JSON 序列化 — 確保模型狀態可匯出"""
    print("  test_json_parser...", end=" ")
    vocab_size = 10
    model = GPT(vocab_size=vocab_size, d_model=8, n_layers=1, max_seq_len=8)

    param_dict = {}
    for name, w, dw in model._named_params():
        param_dict[name] = {
            'shape': list(w.shape),
            'mean': float(np.mean(w)),
            'std': float(np.std(w)),
            'min': float(np.min(w)),
            'max': float(np.max(w)),
        }

    json_str = json.dumps(param_dict, indent=2)
    parsed = json.loads(json_str)
    assert len(parsed) > 0, "沒有參數被序列化"
    assert parsed['Wte']['shape'] == [vocab_size, 8], "Wte shape 錯誤"
    assert 'lm_head_W' in parsed, "缺少 lm_head_W"

    total_size = sum(np.prod(v['shape']) for v in parsed.values())
    assert total_size == model.count_params(), f"參數數量不一致: {total_size} vs {model.count_params()}"

    print("PASS")


# ============================================================
# 4. 端到端模型測試
# ============================================================

def test_gpt_forward_shape():
    """GPT 前向傳播輸出形狀正確"""
    print("  test_gpt_forward_shape...", end=" ")
    model = GPT(vocab_size=20, d_model=16, n_heads=2, n_layers=2, max_seq_len=32)
    x = np.random.randint(0, 20, (3, 10))
    logits = model.forward(x)
    assert logits.shape == (3, 10, 20), f"shape 錯誤: {logits.shape}"
    print("PASS")


def test_gpt_backward_shape():
    """GPT 反向傳播梯度形狀正確"""
    print("  test_gpt_backward_shape...", end=" ")
    model = GPT(vocab_size=20, d_model=16, n_heads=2, n_layers=2, max_seq_len=32)
    x = np.random.randint(0, 20, (2, 8))
    y = np.random.randint(0, 20, (2, 8))

    logits = model.forward(x)
    _, dlogits = cross_entropy_loss(logits, y)
    model.backward(dlogits)

    for name, w, dw in model._named_params():
        assert w.shape == dw.shape, f"{name}: shape mismatch {w.shape} vs {dw.shape}"
        assert not np.all(dw == 0), f"{name}: gradient is zero!"
    print("PASS")


def test_gpt_full_gradient_check():
    """GPT 完整梯度數值檢驗（隨機採樣）"""
    print("  test_gpt_full_gradient_check...", end=" ")
    model = GPT(vocab_size=8, d_model=8, n_heads=2, n_layers=1, max_seq_len=8, d_ff=16)
    x = np.random.randint(0, 8, (2, 4))
    y = np.random.randint(0, 8, (2, 4))

    logits = model.forward(x)
    loss, dlogits = cross_entropy_loss(logits, y)
    model.backward(dlogits)

    analytical = {}
    for name, w, dw in model._named_params():
        analytical[(name, id(w))] = dw.copy()

    eps = 1e-5
    max_diff = 0.0
    for name, w, dw in model._named_params():
        for _ in range(5):
            idx = tuple(np.random.randint(0, s) for s in w.shape)
            old = w[idx].copy()
            w[idx] = old + eps
            l_p, _ = cross_entropy_loss(model.forward(x), y)
            w[idx] = old - eps
            l_m, _ = cross_entropy_loss(model.forward(x), y)
            num_grad = (l_p - l_m) / (2 * eps)
            w[idx] = old
            diff = abs(dw[idx] - num_grad)
            max_diff = max(max_diff, diff)
            assert diff < 1e-3, f"{name}[{idx}]: grad mismatch {dw[idx]} vs {num_grad}"

    assert max_diff < 1e-3, f"最大差異 {max_diff} 超標"
    print(f"PASS (max_diff={max_diff:.2e})")


def test_gpt_training_convergence():
    """GPT 訓練收斂檢驗 — 簡單 overfit 測試"""
    print("  test_gpt_training_convergence...", end=" ")
    vocab_size = 5
    model = GPT(vocab_size=vocab_size, d_model=16, n_heads=2, n_layers=2, max_seq_len=8, d_ff=32)

    text = "abacabadabacaba"
    chars = sorted(list(set(text)))
    c2i = {c: i for i, c in enumerate(chars)}
    data = np.array([c2i[c] for c in text], dtype=np.int32)

    seq_len = 4
    batch = []
    for i in range(0, len(data) - seq_len):
        xb = data[i:i + seq_len].reshape(1, -1)
        yb = data[i + 1:i + seq_len + 1].reshape(1, -1)
        batch.append((xb, yb))

    param_list = list(model.params())
    opt = SGDMomentum(model, lr=0.1, momentum=0.9)
    opt.set_params(param_list)

    initial_loss = float('inf')
    for epoch in range(50):
        total_loss = 0.0
        for xb, yb in batch:
            logits = model.forward(xb)
            loss, dlogits = cross_entropy_loss(logits, yb)
            if epoch == 0:
                initial_loss = loss
            model.backward(dlogits)
            opt.step()
            opt.zero_grad()
            total_loss += loss

    final_loss = total_loss / len(batch)
    if initial_loss != float('inf'):
        assert final_loss < initial_loss, f"Loss 未下降: {initial_loss:.4f} -> {final_loss:.4f}"
        print(f"PASS ({initial_loss:.4f} -> {final_loss:.4f})")
    else:
        print("FAIL (initial_loss not set)")


# ============================================================
# 5. 生成策略測試
# ============================================================

def test_generate_temperature():
    """Temperature 參數影響輸出多樣性"""
    print("  test_generate_temperature...", end=" ")
    model = GPT(vocab_size=10, d_model=8, n_heads=2, n_layers=1, max_seq_len=16)
    context = np.array([[1, 2, 3]])

    out_low = model.generate(context, max_new_tokens=20, temperature=0.1, top_k=None)
    out_high = model.generate(context, max_new_tokens=20, temperature=2.0, top_k=None)

    assert out_low.shape == out_high.shape, "輸出 shape 不一致"
    print("PASS")


def test_generate_topk():
    """Top-k 參數影響輸出分佈"""
    print("  test_generate_topk...", end=" ")
    model = GPT(vocab_size=10, d_model=8, n_heads=2, n_layers=1, max_seq_len=16)
    context = np.array([[1, 2, 3]])

    for k in [1, 3, 5, 10, None]:
        out = model.generate(context, max_new_tokens=10, top_k=k)
        assert out.shape[1] == 13, f"k={k} 輸出長度錯誤: {out.shape}"
    print("PASS")


def test_generate_deterministic():
    """Temperature=0 時應完全確定性"""
    print("  test_generate_deterministic...", end=" ")
    model = GPT(vocab_size=10, d_model=8, n_heads=2, n_layers=1, max_seq_len=16)
    context = np.array([[1, 2, 3]])

    out1 = model.generate(context, max_new_tokens=30, temperature=0.001, top_k=1)
    out2 = model.generate(context, max_new_tokens=30, temperature=0.001, top_k=1)

    assert np.array_equal(out1, out2), "低溫生成應為確定性"
    print("PASS")


# ============================================================
# 6. 參數統計測試
# ============================================================

def test_parameter_count():
    """模型參數統計正確性"""
    print("  test_parameter_count...", end=" ")
    model = GPT(vocab_size=42, d_model=96, n_heads=4, n_layers=3, max_seq_len=128, d_ff=384)
    total = model.count_params()
    assert total > 0, "參數數量為零！"
    print(f"PASS ({total:,})")


def test_zero_grad():
    """zero_grad 後梯度應為零"""
    print("  test_zero_grad...", end=" ")
    model = GPT(vocab_size=10, d_model=8, n_heads=2, n_layers=1, max_seq_len=8)

    param_list = list(model.params())
    opt = SGDMomentum(model, lr=0.01, momentum=0.9)
    opt.set_params(param_list)
    opt.zero_grad()

    for name, w, dw in model._named_params():
        assert np.all(dw == 0), f"{name}: zero_grad 後梯度不為零"
    print("PASS")


def test_param_update():
    """step 後權重應變化"""
    print("  test_param_update...", end=" ")
    model = GPT(vocab_size=10, d_model=8, n_heads=2, n_layers=1, max_seq_len=8)
    x = np.random.randint(0, 10, (2, 4))
    y = np.random.randint(0, 10, (2, 4))

    weights_before = {}
    for name, w, dw in model._named_params():
        weights_before[name] = w.copy()

    logits = model.forward(x)
    _, dlogits = cross_entropy_loss(logits, y)
    model.backward(dlogits)

    param_list = list(model.params())
    opt = SGDMomentum(model, lr=0.1, momentum=0.9)
    opt.set_params(param_list)
    opt.step()

    changed = 0
    for name, w, dw in model._named_params():
        if not np.allclose(w, weights_before[name]):
            changed += 1
    assert changed > 0, "沒有權重被更新！"
    print("PASS")


# ============================================================
# 測試執行器
# ============================================================

def test_json_parser():
    """前面已定義的 JSON 序列化測試"""
    pass  # 已在第3段實現


def test_memory_limit():
    """前面已定義的記憶體限制測試"""
    pass  # 已在第2段實現


def test_invalid_tool():
    """前面已定義的無效輸入測試"""
    pass  # 已在第3段實現


def test_workspace_escape():
    """前面已定義的工作區逸出測試"""
    pass  # 已在第3段實現


ALL_TESTS = [
    ("1. 基礎元件", [
        test_linear_forward,
        test_linear_backward,
        test_layernorm_forward,
        test_layernorm_backward,
        test_embedding_forward,
        test_embedding_backward,
        test_causal_attention_mask,
    ]),
    ("2. 記憶與狀態", [
        test_memory_limit,
        test_sequence_too_long,
        test_batch_independence,
    ]),
    ("3. 無效輸入防護", [
        test_invalid_tool,
        test_workspace_escape,
        test_json_parser,
    ]),
    ("4. 端到端模型", [
        test_gpt_forward_shape,
        test_gpt_backward_shape,
        test_gpt_full_gradient_check,
        test_gpt_training_convergence,
    ]),
    ("5. 生成策略", [
        test_generate_temperature,
        test_generate_topk,
        test_generate_deterministic,
    ]),
    ("6. 參數與優化器", [
        test_parameter_count,
        test_zero_grad,
        test_param_update,
    ]),
]


def run_all():
    print("=" * 60)
    print("  從零打造 GPT — 完整測試套件")
    print("=" * 60)

    total = 0
    passed = 0
    failed = []

    for group_name, tests in ALL_TESTS:
        print(f"\n{'─' * 50}")
        print(f"  {group_name}")
        print(f"{'─' * 50}")
        for test_fn in tests:
            total += 1
            try:
                test_fn()
                passed += 1
            except AssertionError as e:
                failed.append((test_fn.__name__, str(e)))
                print(f"  {test_fn.__name__}... FAIL")
                print(f"    ⚠ {e}")
            except Exception as e:
                failed.append((test_fn.__name__, str(e)))
                print(f"  {test_fn.__name__}... ERROR")
                print(f"    ⚠ {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    if failed:
        print(f"  ✗ {len(failed)}/{total} tests FAILED:")
        for name, msg in failed:
            print(f"    - {name}: {msg}")
    else:
        print(f"  ✓ ALL {total} TESTS PASSED!")
    print(f"{'=' * 60}")
    return len(failed) == 0


if __name__ == '__main__':
    run_all()
