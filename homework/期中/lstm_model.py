import torch
import torch.nn as nn
import numpy as np
import argparse
import os
from typing import Optional

# ==========================================
# 1. 語言模型架構 (Language Model)
# ==========================================
class LSTMLanguageModelAdvanced(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
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

class TextTokenizer:
    def __init__(self, text=None):
        if text is not None: self.fit(text)
    def fit(self, text):
        self.chars = sorted(list(set(text)))
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.i2c = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    def encode(self, text): return [self.c2i[c] for c in text if c in self.c2i]
    def decode(self, ids): return ''.join(self.i2c[i] for i in ids)
    def save(self, path): torch.save({'chars': ''.join(self.chars), 'c2i': self.c2i}, path)
    def load(self, path):
        data = torch.load(path, map_location='cpu')
        self.chars = list(data['chars'])
        self.c2i = data['c2i']
        self.i2c = {int(v): k for k, v in data['c2i'].items()}
        self.vocab_size = len(self.chars)

def load_shakespeare():
    return "O Romeo, Romeo! wherefore art thou Romeo?\nDeny thy father and refuse thy name;\nOr, if thou wilt not, be but sworn my love,\nAnd I'll no longer be a Capulet.\n"

# ==========================================
# 2. 機器控制模型與數據生成 (Robot Control)
# ==========================================
class RobotControllerLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=2, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x, hidden=None):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

def generate_robot_data(num_trajectories=100, seq_len=40):
    X, Y = [], []
    for _ in range(num_trajectories):
        current_pos = np.random.uniform(-5, 5, 2)
        current_vel = np.zeros(2)
        target_pos = np.random.uniform(-10, 10, 2)
        traj_X, traj_Y = [], []
        dt, kp, kd = 0.1, 1.2, 0.5
        for _ in range(seq_len):
            state = np.hstack([current_pos, current_vel, target_pos])
            traj_X.append(state)
            error = target_pos - current_pos
            control_acc = np.clip(kp * error - kd * current_vel, -3.0, 3.0)
            traj_Y.append(control_acc)
            current_pos += current_vel * dt + 0.5 * control_acc * (dt**2)
            current_vel += control_acc * dt
        X.append(traj_X); Y.append(traj_Y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ==========================================
# 3. 啟動命令邏輯 (CLI Commands)
# ==========================================
def cmd_control(args):
    print("\n=== 啟動 V3 改良版：機器人運動軌跡控制引擎 ===")
    X, Y = generate_robot_data(num_trajectories=200, seq_len=40)
    model = RobotControllerLSTM(input_size=6, hidden_size=64, output_size=2, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output, _ = model(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}: 軌跡控制預測誤差 (MSE Loss) = {loss.item():.5f}")
            
    model.eval()
    with torch.no_grad():
        print("\n[控制訊號即時模擬測試] 任務：自 (0,0) 前往 (5,5)")
        current_state = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 5.0, 5.0]]], dtype=torch.float32)
        hidden = None
        for step in range(5):
            pred, hidden = model(current_state, hidden)
            ax, ay = pred[0, -1, 0].item(), pred[0, -1, 1].item()
            print(f"  時間步 {step+1}: 輸出控制訊號 -> 加速度 ax: {ax:.3f}, ay: {ay:.3f}")
            current_state = torch.tensor([[[1.0*(step+1), 1.0*(step+1), ax, ay, 5.0, 5.0]]], dtype=torch.float32)

def cmd_train(args):
    print("\n=== 啟動語言模型訓練 ===")
    text = load_shakespeare()
    tokenizer = TextTokenizer(text)
    model = LSTMLanguageModelAdvanced(vocab_size=tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs('.', exist_ok=True)
    tokenizer.save('checkpoint.pt.tok')
    
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    x = data[:-1].unsqueeze(0)
    y = data[1:].unsqueeze(0)
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        logits, _ = model(x)
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}: Loss={loss.item():.4f}")
            
    torch.save({'model_state_dict': model.state_dict()}, 'checkpoint.pt')
    print("  語言模型訓練完成並已儲存權重。")

def cmd_chat(args):
    tokenizer = TextTokenizer()
    if not os.path.exists('checkpoint.pt.tok'):
        print("找不到字典檔，請先執行 train 指令。"); return
    tokenizer.load('checkpoint.pt.tok')
    model = LSTMLanguageModelAdvanced(vocab_size=tokenizer.vocab_size)
    checkpoint = torch.load('checkpoint.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nLSTM 語言模型互動對話模式 (輸入 quit 離開)")
    while True:
        prompt = input("\n>> ")
        if prompt.strip().lower() in ('quit', 'exit'): break
        if not prompt.strip(): continue
        prompt_ids = tokenizer.encode(prompt)
        out_ids = model.generate(prompt_ids, max_new_tokens=50)
        print(tokenizer.decode(out_ids))

def main():
    parser = argparse.ArgumentParser(description='PyTorch 多功能神經網路引擎')
    sub = parser.add_subparsers(dest='cmd', required=True)
    
    # 1. 機器控制功能指令
    p_control = sub.add_parser('control', help='啟動機器控制實驗')
    p_control.add_argument('--epochs', type=int, default=30)
    
    # 2. 語言模型訓練指令
    p_train = sub.add_parser('train', help='啟動語言模型訓練')
    p_train.add_argument('--epochs', type=int, default=100)
    
    # 3. 語言模型聊天指令
    p_chat = sub.add_parser('chat', help='啟動對話接龍模式')
    
    args = parser.parse_args()
    if args.cmd == 'control': cmd_control(args)
    elif args.cmd == 'train': cmd_train(args)
    elif args.cmd == 'chat': cmd_chat(args)

if __name__ == '__main__':
    main()