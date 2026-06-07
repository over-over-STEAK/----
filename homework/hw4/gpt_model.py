import numpy as np
from typing import Tuple, List, Optional


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.dgamma = np.zeros(dim)
        self.dbeta = np.zeros(dim)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x - mean) * std_inv
        out = self.gamma * x_hat + self.beta
        self.cache = {'x': x, 'x_hat': x_hat, 'mean': mean, 'std_inv': std_inv}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache['x']
        x_hat = self.cache['x_hat']
        mean = self.cache['mean']
        std_inv = self.cache['std_inv']
        N = x.shape[-1]

        dL_dx_hat = dout * self.gamma
        dvar = np.sum(dL_dx_hat * (x - mean) * (-0.5) * std_inv ** 3, axis=-1, keepdims=True)
        dmu = np.sum(dL_dx_hat * (-std_inv), axis=-1, keepdims=True) + dvar * np.sum(-2 * (x - mean), axis=-1, keepdims=True) / N
        dx = dL_dx_hat * std_inv + dvar * 2 * (x - mean) / N + dmu / N

        self.dgamma = np.sum(dout * x_hat, axis=(0, 1))
        self.dbeta = np.sum(dout, axis=(0, 1))
        return dx

    def params(self):
        return [('gamma', self.gamma, self.dgamma), ('beta', self.beta, self.dbeta)]


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * scale
        self.b = np.zeros(out_features) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.cache['x_shape'] = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        self.cache['x_2d'] = x_2d
        out_2d = x_2d @ self.W.T
        if self.b is not None:
            out_2d += self.b
        return out_2d.reshape(*x.shape[:-1], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x_2d = self.cache['x_2d']
        orig_shape = self.cache['x_shape']
        dout_2d = dout.reshape(-1, dout.shape[-1])

        self.dW = dout_2d.T @ x_2d
        if self.b is not None:
            self.db = np.sum(dout_2d, axis=0)

        dx_2d = dout_2d @ self.W
        return dx_2d.reshape(orig_shape)

    def params(self):
        p = [('W', self.W, self.dW)]
        if self.b is not None:
            p.append(('b', self.b, self.db))
        return p


class Embedding:
    def __init__(self, vocab_size: int, dim: int):
        scale = np.sqrt(2.0 / dim)
        self.W = np.random.randn(vocab_size, dim) * scale
        self.dW = np.zeros_like(self.W)
        self.cache = {}

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        self.cache['input_ids'] = input_ids
        return self.W[input_ids]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        np.add.at(self.dW, self.cache['input_ids'], dout)
        return None

    def params(self):
        return [('Wte', self.W, self.dW)]


class CausalSelfAttention:
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = Linear(d_model, d_model, bias=False)
        self.W_K = Linear(d_model, d_model, bias=False)
        self.W_V = Linear(d_model, d_model, bias=False)
        self.W_O = Linear(d_model, d_model, bias=False)

        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        H = self.n_heads
        d_k = self.d_k

        Q = self.W_Q.forward(x)
        K = self.W_K.forward(x)
        V = self.W_V.forward(x)

        Q = Q.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(d_k)

        mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), k=1)
        scores = scores + mask

        A = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        A = A / np.sum(A, axis=-1, keepdims=True)

        O = A @ V

        O_combined = O.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = self.W_O.forward(O_combined)

        self.cache = {
            'x': x, 'Q': Q, 'K': K, 'V': V, 'A': A, 'O': O,
            'O_combined': O_combined, 'B': B, 'T': T, 'D': D, 'H': H, 'd_k': d_k
        }
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self.cache['x']
        Q = self.cache['Q']
        K = self.cache['K']
        V = self.cache['V']
        A = self.cache['A']
        O = self.cache['O']
        O_combined = self.cache['O_combined']
        B, T, D = self.cache['B'], self.cache['T'], self.cache['D']
        H, d_k = self.cache['H'], self.cache['d_k']

        dO_combined = self.W_O.backward(dout)

        dO = dO_combined.reshape(B, T, H, d_k).transpose(0, 2, 1, 3)

        dA = dO @ V.transpose(0, 1, 3, 2)
        dV = A.transpose(0, 1, 3, 2) @ dO

        dS = A * (dA - np.sum(dA * A, axis=-1, keepdims=True))
        dS = dS / np.sqrt(d_k)

        dQ = dS @ K
        dK = dS.transpose(0, 1, 3, 2) @ Q

        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, T, D)

        dx_Q = self.W_Q.backward(dQ)
        dx_K = self.W_K.backward(dK)
        dx_V = self.W_V.backward(dV)

        return dx_Q + dx_K + dx_V

    def params(self):
        p = []
        p.extend(self.W_Q.params())
        p.extend(self.W_K.params())
        p.extend(self.W_V.params())
        p.extend(self.W_O.params())
        return [(f'attention_{name}', w, dw) for name, w, dw in p]


class FeedForward:
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        if d_ff is None:
            d_ff = 4 * d_model
        self.W_1 = Linear(d_model, d_ff)
        self.W_2 = Linear(d_ff, d_model)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.W_1.forward(x)
        self.cache['h'] = h
        a = np.maximum(h, 0)
        self.cache['a'] = a
        out = self.W_2.forward(a)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        h = self.cache['h']
        a = self.cache['a']

        da = self.W_2.backward(dout)
        dh = da * (h > 0).astype(np.float32)
        dx = self.W_1.backward(dh)
        return dx

    def params(self):
        p = []
        p.extend(self.W_1.params())
        p.extend(self.W_2.params())
        return [(f'ffn_{name}', w, dw) for name, w, dw in p]


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: Optional[int] = None):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # FFN residual path
        dx_ffn = self.ffn.backward(dout)
        dx_ln2 = self.ln2.backward(dx_ffn)
        dx = dout + dx_ln2

        # Attention residual path
        dx_attn = self.attn.backward(dx)
        dx_ln1 = self.ln1.backward(dx_attn)
        dx = dx + dx_ln1
        return dx

    def params(self):
        p = []
        for name, w, dw in self.ln1.params():
            p.append((f'ln1_{name}', w, dw))
        for name, w, dw in self.attn.params():
            p.append((name, w, dw))
        for name, w, dw in self.ln2.params():
            p.append((f'ln2_{name}', w, dw))
        for name, w, dw in self.ffn.params():
            p.append((name, w, dw))
        return p


class GPT:
    def __init__(self, vocab_size: int, d_model: int = 96, n_heads: int = 4,
                 n_layers: int = 3, max_seq_len: int = 256, d_ff: Optional[int] = None):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.wte = Embedding(vocab_size, d_model)
        self.wpe = Embedding(max_seq_len, d_model)

        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size, bias=False)

        self.cache = {}
        self._init_params()

    def _init_params(self):
        pass

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        B, T = input_ids.shape
        pos = np.arange(T, dtype=np.int32)

        token_emb = self.wte.forward(input_ids)
        pos_emb = self.wpe.forward(pos)
        x = token_emb + pos_emb

        for block in self.blocks:
            x = block.forward(x)

        x = self.ln_f.forward(x)
        logits = self.lm_head.forward(x)
        self.cache['logits'] = logits
        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        dx = self.lm_head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for block in reversed(self.blocks):
            dx = block.backward(dx)

    def generate(self, input_ids: np.ndarray, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None) -> np.ndarray:
        B = input_ids.shape[0]
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.max_seq_len:
                context = input_ids[:, -self.max_seq_len:]
            else:
                context = input_ids

            logits = self.forward(context)
            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None:
                top_k_vals = np.sort(next_logits)[:, -top_k]
                next_logits = np.where(next_logits >= top_k_vals[:, None], next_logits, -np.inf)

            probs = np.exp(next_logits - np.max(next_logits, axis=-1, keepdims=True))
            probs = probs / np.sum(probs, axis=-1, keepdims=True)

            next_token = np.array([np.random.choice(self.vocab_size, p=probs[0])])
            input_ids = np.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)
        return input_ids

    def count_params(self) -> int:
        total = 0
        for name, w, dw in self._named_params():
            total += w.size
        return total

    def _named_params(self):
        for name, w, dw in self.wte.params():
            yield (name, w, dw)
        for name, w, dw in self.wpe.params():
            yield (name, w, dw)
        for i, block in enumerate(self.blocks):
            for name, w, dw in block.params():
                yield (f'block_{i}_{name}', w, dw)
        for name, w, dw in self.ln_f.params():
            yield (f'ln_f_{name}', w, dw)
        for name, w, dw in self.lm_head.params():
            yield (f'lm_head_{name}', w, dw)

    def params(self):
        for _, w, dw in self._named_params():
            yield (w, dw)

    def param_count_by_layer(self) -> dict:
        counts = {}
        for name, w, _ in self._named_params():
            layer = name.split('_')[0] if '_' in name else name
            counts[name] = w.size
        return counts


class SGDMomentum:
    def __init__(self, model: GPT, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
        for i, (w, dw) in enumerate(model.params()):
            self.velocities[id(w)] = np.zeros_like(w)

    def step(self):
        for w, dw in self.model_params:
            v = self.velocities[id(w)]
            v[:] = self.momentum * v + self.lr * dw
            w[:] -= v[:]

    def set_params(self, model_param_list):
        self.model_params = model_param_list

    def zero_grad(self):
        for w, dw in self.model_params:
            dw[:] = 0


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    B, T, V = logits.shape
    logits_2d = logits.reshape(-1, V)
    targets_1d = targets.reshape(-1)

    max_vals = np.max(logits_2d, axis=-1, keepdims=True)
    shifted = logits_2d - max_vals
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)

    N = probs.shape[0]
    probs_clipped = np.clip(probs, 1e-15, 1.0)
    loss = -np.mean(np.log(probs_clipped[np.arange(N), targets_1d]))

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), targets_1d] = 1
    dlogits_2d = (probs - one_hot) / N
    dlogits = dlogits_2d.reshape(B, T, V)
    return loss, dlogits
