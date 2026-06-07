import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from typing import Optional


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden


class LSTMLanguageModelAdvanced(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size=128,
                 hidden_size=256,
                 num_layers=2,
                 dropout=0.3,
                 tie_weights=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_weights = tie_weights

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.proj_in = nn.Identity() if embed_size == hidden_size else nn.Linear(embed_size, hidden_size)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Identity() if embed_size == hidden_size else nn.Linear(hidden_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)

        if tie_weights:
            self.fc.weight = self.embedding.weight

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x = self.proj_in(x)
        x = self.dropout(x)
        output, hidden = self.lstm(x, hidden)
        output = self.dropout(output)
        output = self.proj_out(output)
        output = self.fc(output)
        return output, hidden

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        self.eval()
        device = next(self.parameters()).device
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        _, hidden = self.forward(ids)
        result = prompt_ids[:]

        x = ids[:, -1:]
        for _ in range(max_new_tokens):
            logits, hidden = self.forward(x, hidden)
            logits = logits[:, -1, :].squeeze(0)

            if temperature < 1e-6:
                next_id = torch.argmax(logits).item()
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    k = min(top_k, self.vocab_size)
                    vals, _ = torch.topk(logits, k)
                    logits[logits < vals[-1]] = -float('inf')
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            result.append(next_id)
            x = torch.tensor([[next_id]], dtype=torch.long, device=device)

        return result

    def count_params(self):
        return sum(p.numel() for p in self.parameters())



class TextTokenizer:
    def __init__(self, text=None):
        if text is not None:
            self.fit(text)

    def fit(self, text):
        self.chars = sorted(list(set(text)))
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.c2i[c] for c in text if c in self.c2i]

    def decode(self, ids):
        return ''.join(self.i2c[i] for i in ids)

    def save(self, path):
        data = {
            'chars': ''.join(self.chars),
            'c2i': self.c2i,
        }
        torch.save(data, path)

    def load(self, path):
        data = torch.load(path, map_location='cpu')
        self.chars = list(data['chars'])
        self.c2i = data['c2i']
        # 修正原先程式碼中試圖將字元轉 int 的 bug，正確對調 key 和 value
        self.i2c = {int(v): k for k, v in data['c2i'].items()}
        self.vocab_size = len(self.chars)


def load_shakespeare():
    return (
        "O Romeo, Romeo! wherefore art thou Romeo?\n"
        "Deny thy father and refuse thy name;\n"
        "Or, if thou wilt not, be but sworn my love,\n"
        "And I'll no longer be a Capulet.\n"
        "'Tis but thy name that is my enemy;\n"
        "Thou art thyself, though not a Montague.\n"
        "What's Montague? it is nor hand, nor foot,\n"
        "Nor arm, nor face, nor any other part\n"
        "Belonging to a man. O, be some other name!\n"
        "What's in a name? that which we call a rose\n"
        "By any other name would smell as sweet;\n"
        "So Romeo would, were he not Romeo call'd,\n"
        "Retain that dear perfection which he owes\n"
        "Without that title. Romeo, doff thy name,\n"
        "And for that name which is no part of thee\n"
        "Take all myself.\n"
    )


def get_batches(data, batch_size, seq_len):
    n = len(data)
    stride = seq_len // 2
    total = (n - 1) // (batch_size * stride) * (batch_size * stride)
    if total == 0:
        total = (n - 1) // seq_len * seq_len
        x = data[:total].reshape(1, -1)
    else:
        x = data[:total].reshape(batch_size, -1)
    y = data[1:total + 1].reshape(x.shape)
    batches = []
    for i in range(0, x.shape[1] - seq_len + 1, stride):
        batches.append((x[:, i:i+seq_len], y[:, i:i+seq_len]))
    return batches


class Trainer:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.history = []

    def train(self, tokenizer, text, epochs=60, batch_size=8, seq_len=32,
              lr=0.005, weight_decay=1e-4, clip=1.0, lr_step=40, lr_gamma=0.5,
              log_every=10, save_path=None):
        self.model.train()
        data = np.array(tokenizer.encode(text), dtype=np.int64)
        batches = get_batches(data, batch_size, seq_len)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
        criterion = nn.CrossEntropyLoss()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            tokenizer.save(save_path + '.tok')

        print(f"訓練：{len(batches)} batches × {epochs} epochs | device={self.device}")
        for epoch in range(epochs):
            np.random.shuffle(batches)
            total_loss = 0
            for xb, yb in batches:
                x = torch.tensor(xb, dtype=torch.long, device=self.device)
                y = torch.tensor(yb, dtype=torch.long, device=self.device)

                logits, _ = self.model(x)
                loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                if clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / max(len(batches), 1)
            self.history.append(avg_loss)

            if (epoch + 1) % log_every == 0:
                ppl = np.exp(avg_loss)
                lr_now = optimizer.param_groups[0]['lr']
                print(f"  epoch {epoch+1:>3d}: loss={avg_loss:.4f}  ppl={ppl:.2f}  lr={lr_now:.5f}")

            if save_path and ((epoch + 1) % 20 == 0 or epoch == epochs - 1):
                self.save(save_path)

        return self.history

    @torch.no_grad()
    def generate(self, tokenizer, prompt, max_new_tokens=100, temperature=1.0, top_k=None):
        self.model.eval()
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [0]
        out = self.model.generate(prompt_ids, max_new_tokens, temperature, top_k)
        return tokenizer.decode(out)

    def save(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
        }, path)
        print(f"  已儲存 {path}")

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        if isinstance(data, dict) and 'model_state_dict' in data:
            self.model.load_state_dict(data['model_state_dict'])
            self.history = data.get('history', [])
        else:
            self.model.load_state_dict(data)
        print(f"  已載入 {path}")
        return self


def cmd_train(args):
    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = load_shakespeare()

    tokenizer = TextTokenizer(text)
    model = LSTMLanguageModelAdvanced(
        vocab_size=tokenizer.vocab_size,
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    )

    save_path = args.save or 'checkpoint.pt'
    trainer = Trainer(model)
    trainer.train(
        tokenizer=tokenizer,
        text=text,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        log_every=args.log_every,
        save_path=save_path,
    )

    print(f"\n訓練完成！參數量：{model.count_params():,}")
    for temp, topk in [(0.5, 15), (0.8, 10), (1.0, None)]:
        out = trainer.generate(tokenizer, "O Romeo, Romeo!", 80, temp, topk)
        print(f"\n  T={temp}, top_k={topk}: {out[:100]}")


def cmd_generate(args):
    tok_path = args.checkpoint + '.tok'
    if os.path.exists(tok_path):
        tokenizer = TextTokenizer()
        tokenizer.load(tok_path)
    else:
        print("找不到 tokenizer 檔案，請先執行訓練 (train)。")
        return

    model = LSTMLanguageModelAdvanced(
        vocab_size=tokenizer.vocab_size,
        embed_size=64,
        hidden_size=128
    )
    trainer = Trainer(model)
    trainer.load(args.checkpoint)

    out = trainer.generate(tokenizer, args.prompt, args.length, args.temperature, args.top_k)
    print("\n--- 生成結果 ---")
    print(out)


def cmd_chat(args):
    tok_path = args.checkpoint + '.tok'
    if os.path.exists(tok_path):
        tokenizer = TextTokenizer()
        tokenizer.load(tok_path)
    else:
        print("找不到 tokenizer 檔案，請先執行訓練 (train)。")
        return

    model = LSTMLanguageModelAdvanced(
        vocab_size=tokenizer.vocab_size,
        embed_size=64,
        hidden_size=128
    )
    trainer = Trainer(model)
    trainer.load(args.checkpoint)

    print("\nLSTM 互動對話模式 (輸入 quit 或 exit 離開)")
    while True:
        try:
            prompt = input("\n>> ")
            if prompt.strip().lower() in ('quit', 'exit'):
                break
            if not prompt.strip():
                continue
            out = trainer.generate(tokenizer, prompt, args.length, args.temperature, args.top_k)
            print(out)
        except (KeyboardInterrupt, EOFError):
            break


def main():
    parser = argparse.ArgumentParser(description='PyTorch LSTM Language Model')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # 訓練命令參數
    p_train = sub.add_parser('train')
    p_train.add_argument('--file', type=str, help='訓練用文字檔路徑 (選填)')
    p_train.add_argument('--epochs', type=int, default=100)
    p_train.add_argument('--batch-size', type=int, default=8)
    p_train.add_argument('--seq-len', type=int, default=32)
    p_train.add_argument('--embed-size', type=int, default=64)
    p_train.add_argument('--hidden-size', type=int, default=128)
    p_train.add_argument('--num-layers', type=int, default=2)
    p_train.add_argument('--dropout', type=float, default=0.3)
    p_train.add_argument('--lr', type=float, default=0.005)
    p_train.add_argument('--lr-step', type=int, default=30)
    p_train.add_argument('--lr-gamma', type=float, default=0.5)
    p_train.add_argument('--tie-weights', action='store_true')
    p_train.add_argument('--save', type=str, default='checkpoint.pt')
    p_train.add_argument('--log-every', type=int, default=10)
    
    p_gen = sub.add_parser('generate')
    p_gen.add_argument('--checkpoint', type=str, required=True)
    p_gen.add_argument('--prompt', type=str, default='O Romeo, Romeo!')
    p_gen.add_argument('--length', type=int, default=200)
    p_gen.add_argument('--temperature', type=float, default=0.8)
    p_gen.add_argument('--top-k', type=int, default=15)

    p_chat = sub.add_parser('chat')
    p_chat.add_argument('--checkpoint', type=str, required=True)
    p_chat.add_argument('--length', type=int, default=150)
    p_chat.add_argument('--temperature', type=float, default=0.8)
    p_chat.add_argument('--top-k', type=int, default=10)

    args = parser.parse_args()

    if args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'generate':
        cmd_generate(args)
    elif args.cmd == 'chat':
        cmd_chat(args)


if __name__ == '__main__':
    main()