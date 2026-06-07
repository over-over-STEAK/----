import numpy as np
import math
import random

np.random.seed(2026)

# ============================================================
# GPT — 從零實作 (Pure NumPy)
# 包含: Embedding, Positional Encoding, Causal Multi-Head Attention,
#       Feed-Forward, LayerNorm, Transformer Block, 訓練與生成
# ============================================================

# -----------------------------------------------------------
# 1. Layer Normalization
# -----------------------------------------------------------
class LayerNorm:
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        orig_shape = dout.shape
        if dout.ndim == 3:
            B, T, D = dout.shape
            dout = dout.reshape(-1, D)
            x_hat = self.x_hat.reshape(-1, D)
        else:
            B, T, D = 1, 1, dout.shape[-1]
            x_hat = self.x_hat

        N, Dd = dout.shape
        dgamma = (dout * x_hat).sum(axis=0)
        dbeta = dout.sum(axis=0)
        dx_hat = dout * self.gamma
        dx = (1.0 / N) * (N * dx_hat - dx_hat.sum(axis=0, keepdims=True)
                         - x_hat * (dx_hat * x_hat).sum(axis=0, keepdims=True))
        dx = dx.reshape(orig_shape)
        return dx, dgamma, dbeta


# -----------------------------------------------------------
# 2. Embedding
# -----------------------------------------------------------
class Embedding:
    def __init__(self, vocab_size, d_model):
        self.W = np.random.randn(vocab_size, d_model) * 0.01

    def __call__(self, indices):
        self.indices = indices
        return self.W[indices]

    def backward(self, dout):
        dW = np.zeros_like(self.W)
        np.add.at(dW, self.indices, dout)
        return dW


# -----------------------------------------------------------
# 3. Positional Encoding (Sinusoidal)
# -----------------------------------------------------------
def sinusoidal_pos_encoding(max_len, d_model):
    pos = np.arange(max_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1.0 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    pos_enc = np.zeros((max_len, d_model))
    pos_enc[:, 0::2] = np.sin(pos * angle_rates[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos * angle_rates[:, 1::2])
    return pos_enc


# -----------------------------------------------------------
# 4. Causal Multi-Head Self-Attention (NumPy)
# -----------------------------------------------------------
class CausalMultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.Wq = np.random.randn(d_model, d_model) * 0.02
        self.Wk = np.random.randn(d_model, d_model) * 0.02
        self.Wv = np.random.randn(d_model, d_model) * 0.02
        self.Wo = np.random.randn(d_model, d_model) * 0.02

    def split_heads(self, x, T):
        return x.reshape(-1, T, self.num_heads, self.depth).transpose(0, 2, 1, 3)

    def combine_heads(self, x, T):
        return x.transpose(0, 2, 1, 3).reshape(-1, T, self.d_model)

    def __call__(self, x, T):
        B = x.shape[0]
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv

        Q = self.split_heads(Q, T)
        K = self.split_heads(K, T)
        V = self.split_heads(V, T)

        scale = self.depth ** -0.5
        attn_logits = Q @ K.transpose(0, 1, 3, 2) * scale

        # Causal mask (上三角 = -inf)
        mask = np.triu(np.full((T, T), -1e9), k=1)
        attn_logits += mask

        self.attn_weights = np.exp(attn_logits - attn_logits.max(axis=-1, keepdims=True))
        self.attn_weights /= self.attn_weights.sum(axis=-1, keepdims=True)

        context = self.attn_weights @ V
        context = self.combine_heads(context, T)
        out = context @ self.Wo
        self.x, self.Q, self.K, self.V = x, Q, K, V
        return out

    def backward(self, dout, T):
        B = dout.shape[0]
        dWo = self.combine_heads(self.attn_weights @ self.V, T).transpose(0, 2, 1) @ dout.reshape(-1, T, self.d_model)
        dWo = dWo.sum(axis=0) if dWo.ndim > 2 else dWo
        dcontext = dout @ self.Wo.T
        dcontext = self.split_heads(dcontext, T)

        dV = self.attn_weights.transpose(0, 1, 3, 2) @ dcontext
        dattn = dcontext @ self.V.transpose(0, 1, 3, 2)

        # Softmax backward
        dS = self.attn_weights * (dattn - (dattn * self.attn_weights).sum(axis=-1, keepdims=True))

        scale = self.depth ** -0.5
        dQ = dS @ self.K * scale
        dK = dS.transpose(0, 1, 3, 2) @ self.Q * scale

        dQ = self.combine_heads(dQ, T)
        dK = self.combine_heads(dK, T)
        dV = self.combine_heads(dV, T)

        dx = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        dWq = self.x.reshape(-1, self.d_model).T @ dQ.reshape(-1, self.d_model)
        dWk = self.x.reshape(-1, self.d_model).T @ dK.reshape(-1, self.d_model)
        dWv = self.x.reshape(-1, self.d_model).T @ dV.reshape(-1, self.d_model)

        self.dWq, self.dWk, self.dWv, self.dWo = dWq, dWk, dWv, dWo
        return dx


# -----------------------------------------------------------
# 5. Feed-Forward Network
# -----------------------------------------------------------
class FeedForward:
    def __init__(self, d_model, dff):
        self.W1 = np.random.randn(d_model, dff) * 0.02
        self.b1 = np.zeros(dff)
        self.W2 = np.random.randn(dff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def __call__(self, x):
        self.x = x
        self.h = x @ self.W1 + self.b1
        self.a = np.maximum(self.h, 0)  # ReLU
        return self.a @ self.W2 + self.b2

    def backward(self, dout):
        dW2 = self.a.reshape(-1, self.a.shape[-1]).T @ dout.reshape(-1, dout.shape[-1])
        db2 = dout.sum(axis=(0, 1)) if dout.ndim == 3 else dout.sum(axis=0)
        da = dout @ self.W2.T
        dh = da * (self.h > 0)
        dW1 = self.x.reshape(-1, self.x.shape[-1]).T @ dh.reshape(-1, dh.shape[-1])
        db1 = dh.sum(axis=(0, 1)) if dh.ndim == 3 else dh.sum(axis=0)
        dx = dh @ self.W1.T
        self.dW1, self.db1, self.dW2, self.db2 = dW1, db1, dW2, db2
        return dx


# -----------------------------------------------------------
# 6. Transformer Block
# -----------------------------------------------------------
class TransformerBlock:
    def __init__(self, d_model, num_heads, dff):
        self.attn = CausalMultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, dff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def __call__(self, x, T):
        self.x1 = x
        a = self.ln1(x)
        a = self.attn(a, T)
        x = x + a
        self.x2 = x
        b = self.ln2(x)
        b = self.ffn(b)
        return x + b

    def backward(self, dout, T):
        dx = dout
        db = self.ffn.backward(dx)
        db, dgamma2, dbeta2 = self.ln2.backward(db)
        dx2 = db
        dx = dx2 + dx
        da = self.attn.backward(dx, T)
        da, dgamma1, dbeta1 = self.ln1.backward(da)
        dx1 = da
        dx = dx1 + dx
        self.dgamma1, self.dbeta1 = dgamma1, dbeta1
        self.dgamma2, self.dbeta2 = dgamma2, dbeta2
        return dx


# -----------------------------------------------------------
# 7. GPT Model
# -----------------------------------------------------------
class GPT:
    def __init__(self, vocab_size, max_len, d_model=128, num_heads=4,
                 num_layers=3, dff=256):
        self.max_len = max_len
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embed = Embedding(vocab_size, d_model)
        self.pos_enc = sinusoidal_pos_encoding(max_len, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, dff)
                       for _ in range(num_layers)]
        self.ln_final = LayerNorm(d_model)
        self.head_W = np.random.randn(d_model, vocab_size) * 0.02
        self.head_b = np.zeros(vocab_size)

    def __call__(self, x):
        B, T = x.shape
        h = self.embed(x) * math.sqrt(self.d_model)
        h += self.pos_enc[:T][np.newaxis, :, :]
        for block in self.blocks:
            h = block(h, T)
        h = self.ln_final(h)
        logits = h @ self.head_W + self.head_b
        self.cache = (x, B, T, h)
        return logits

    def loss_and_backward(self, x, y):
        logits = self.__call__(x)
        B, T, V = logits.shape
        logits_flat = logits.reshape(-1, V)
        y_flat = y.reshape(-1)

        # Cross-entropy loss
        logits_max = logits_flat.max(axis=-1, keepdims=True)
        logits_stable = logits_flat - logits_max
        exp_logits = np.exp(logits_stable)
        sum_exp = exp_logits.sum(axis=-1, keepdims=True)
        probs = exp_logits / sum_exp
        loss = -np.mean(np.log(probs[np.arange(len(y_flat)), y_flat] + 1e-10))

        # Grad
        dlogits = probs.copy()
        dlogits[np.arange(len(y_flat)), y_flat] -= 1
        dlogits = dlogits.reshape(B, T, V) / (B * T)

        # Head grad
        h = self.cache[-1]
        dW_head = h.reshape(-1, self.d_model).T @ dlogits.reshape(-1, V)
        db_head = dlogits.sum(axis=(0, 1))

        dh = dlogits @ self.head_W.T
        dh, dgamma, dbeta = self.ln_final.backward(dh)

        for block in reversed(self.blocks):
            dh = block.backward(dh, self.cache[2])

        dW_embed = self.embed.backward(dh)
        self.dW_head, self.db_head = dW_head, db_head
        self.dgamma_final, self.dbeta_final = dgamma, dbeta
        self.dW_embed = dW_embed

        return loss

    def generate(self, start_ids, max_new_tokens=100, temperature=0.8, top_k=20):
        generated = list(start_ids)
        for _ in range(max_new_tokens):
            inp = np.array([generated[-self.max_len:]])
            logits = self.__call__(inp)
            logits = logits[0, -1, :] / temperature

            if top_k > 0:
                indices = np.argpartition(logits, -top_k)[-top_k:]
                mask = np.full_like(logits, -1e9)
                mask[indices] = logits[indices]
                logits = mask

            exp_l = np.exp(logits - logits.max())
            probs = exp_l / exp_l.sum()
            next_id = np.random.choice(len(probs), p=probs)
            generated.append(int(next_id))
        return generated


# -----------------------------------------------------------
# 8. Optimizer (簡單 SGD + Momentum)
# -----------------------------------------------------------
class Optimizer:
    def __init__(self, model, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocities = {}

    def step(self, model):
        params = {
            'embed_W': (model.embed.W, model.dW_embed),
            'head_W': (model.head_W, model.dW_head),
            'head_b': (model.head_b, model.db_head),
            'ln_gamma': (model.ln_final.gamma, model.dgamma_final),
            'ln_beta': (model.ln_final.beta, model.dbeta_final),
        }
        for i, block in enumerate(model.blocks):
            params[f'block{i}_attn_Wq'] = (block.attn.Wq, block.attn.dWq)
            params[f'block{i}_attn_Wk'] = (block.attn.Wk, block.attn.dWk)
            params[f'block{i}_attn_Wv'] = (block.attn.Wv, block.attn.dWv)
            params[f'block{i}_attn_Wo'] = (block.attn.Wo, block.attn.dWo)
            params[f'block{i}_ffn_W1'] = (block.ffn.W1, block.ffn.dW1)
            params[f'block{i}_ffn_b1'] = (block.ffn.b1, block.ffn.db1)
            params[f'block{i}_ffn_W2'] = (block.ffn.W2, block.ffn.dW2)
            params[f'block{i}_ffn_b2'] = (block.ffn.b2, block.ffn.db2)
            params[f'block{i}_ln1_gamma'] = (block.ln1.gamma, block.dgamma1)
            params[f'block{i}_ln1_beta'] = (block.ln1.beta, block.dbeta1)
            params[f'block{i}_ln2_gamma'] = (block.ln2.gamma, block.dgamma2)
            params[f'block{i}_ln2_beta'] = (block.ln2.beta, block.dbeta2)

        for name, (param, grad) in params.items():
            if name not in self.velocities:
                self.velocities[name] = np.zeros_like(param)
            self.velocities[name] = self.beta * self.velocities[name] + (1 - self.beta) * grad
            param -= self.lr * self.velocities[name]


# -----------------------------------------------------------
# 9. Data Preparation
# -----------------------------------------------------------
def load_text():
    text = (
        "In the beginning God created the heaven and the earth. "
        "And the earth was without form and void; and darkness was upon the face of the deep. "
        "And the Spirit of God moved upon the face of the waters. "
        "And God said, Let there be light: and there was light. "
        "And God saw the light, that it was good: and God divided the light from the darkness. "
        "And God called the light Day, and the darkness he called Night. "
        "And the evening and the morning were the first day. "
        "And God said, Let there be a firmament in the midst of the waters, "
        "and let it divide the waters from the waters. "
        "And God made the firmament, and divided the waters which were under the firmament "
        "from the waters which were above the firmament: and it was so. "
        "And God called the firmament Heaven. "
        "And the evening and the morning were the second day. "
        "And God said, Let the waters under the heaven be gathered together unto one place, "
        "and let the dry land appear: and it was so. "
        "And God called the dry land Earth; and the gathering together of the waters called he Seas: "
        "and God saw that it was good. "
        "And God said, Let the earth bring forth grass, the herb yielding seed, "
        "and the fruit tree yielding fruit after his kind, whose seed is in itself, "
        "upon the earth: and it was so."
    )
    return text.lower()


def build_vocab(text):
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return chars, char_to_idx, idx_to_char


def create_batches(text, char_to_idx, seq_length=32, batch_size=16):
    chars = np.array([char_to_idx[c] for c in text])
    n = len(chars)
    Xs, Ys = [], []
    for i in range(0, n - seq_length, 1):
        Xs.append(chars[i:i + seq_length])
        Ys.append(chars[i + 1:i + seq_length + 1])
        if len(Xs) == batch_size:
            yield np.array(Xs), np.array(Ys)
            Xs, Ys = [], []
    if Xs:
        yield np.array(Xs), np.array(Ys)


# -----------------------------------------------------------
# 10. Training
# -----------------------------------------------------------
def train_gpt():
    print("=" * 55)
    print("  GPT -- Generative Pre-trained Transformer")
    print("  Pure NumPy Implementation (No Framework)")
    print("=" * 55)

    text = load_text()
    chars, char_to_idx, idx_to_char = build_vocab(text)
    vocab_size = len(chars)
    print(f"\n[INFO] 詞彙表大小: {vocab_size} 個字元")
    print(f"[INFO] 文本總長度: {len(text)} 字元")
    print(f"[INFO] 字元集合: {''.join(chars)}")

    seq_len = 32
    model = GPT(
        vocab_size=vocab_size,
        max_len=seq_len,
        d_model=96,
        num_heads=4,
        num_layers=3,
        dff=192
    )
    opt = Optimizer(model, lr=0.01)

    total_params = (model.embed.W.size + model.head_W.size + model.head_b.size
                    + model.ln_final.gamma.size + model.ln_final.beta.size)
    for block in model.blocks:
        for p in [block.attn.Wq, block.attn.Wk, block.attn.Wv, block.attn.Wo,
                  block.ffn.W1, block.ffn.b1, block.ffn.W2, block.ffn.b2,
                  block.ln1.gamma, block.ln1.beta, block.ln2.gamma, block.ln2.beta]:
            total_params += p.size
    print(f"\n[INFO] GPT 架構: {model.d_model}d, {len(model.blocks)} layers, {model.blocks[0].attn.num_heads} heads")
    print(f"[INFO] 參數總數: {total_params:,}")

    print(f"\n{'='*55}")
    print("  開始訓練")
    print(f"{'='*55}")

    losses = []
    for epoch in range(1, 31):
        epoch_loss = 0
        n_batches = 0
        for X, Y in create_batches(text, char_to_idx, seq_len, batch_size=16):
            loss = model.loss_and_backward(X, Y)
            opt.step(model)
            epoch_loss += loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f} | Perplexity: {math.exp(avg_loss):.2f}")

    print(f"\n{'='*55}")
    print("  訓練完成！")
    print(f"{'='*55}")

    # --- 文字生成展示 ---
    print(f"\n{'='*55}")
    print("  GPT 文字生成展示")
    print(f"{'='*55}")

    prompts = ["and god said", "in the beginning", "let there be"]
    for prompt in prompts:
        print(f"\n[Prompt] \"{prompt}\"")
        prompt_ids = [char_to_idx.get(c, 0) for c in prompt]
        for temp in [0.3, 0.7, 1.0]:
            generated = model.generate(
                prompt_ids[:],
                max_new_tokens=60,
                temperature=temp,
                top_k=10
            )
            out = ''.join(idx_to_char[i] for i in generated)
            print(f"  temp={temp:.1f}: {out}")

    return model, chars, char_to_idx, idx_to_char


if __name__ == "__main__":
    train_gpt()
