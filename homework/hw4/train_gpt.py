# Wrote gpt-from-scratch\train_gpt.py
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpt_model import GPT, SGDMomentum, cross_entropy_loss


def get_shakespeare_text():
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


def encode_text(text):
    chars = sorted(list(set(text)))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    encoded = np.array([char_to_idx[c] for c in text], dtype=np.int32)
    return encoded, char_to_idx, idx_to_char, len(chars)


def get_batches(data, batch_size, seq_len):
    n = len(data)
    num_batches = (n - 1) // (batch_size * seq_len)

    data = data[:num_batches * batch_size * seq_len + 1]
    x = data[:-1].reshape(batch_size, -1)
    y = data[1:].reshape(batch_size, -1)

    batches = []
    for i in range(0, x.shape[1] - seq_len + 1, seq_len):
        xb = x[:, i:i + seq_len]
        yb = y[:, i:i + seq_len]
        batches.append((xb, yb))

    return batches


def train():
    np.random.seed(42)

    text = get_shakespeare_text()
    data, char_to_idx, idx_to_char, vocab_size = encode_text(text)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Text length: {len(data)} characters")
    print(f"Characters: {''.join(list(char_to_idx.keys()))}")

    d_model = 96
    n_heads = 4
    n_layers = 3
    max_seq_len = 128
    d_ff = 4 * d_model

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        d_ff=d_ff
    )

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model config: {n_layers} layers, {n_heads} heads, {d_model} dim, {d_ff} FF dim")
    print(f"Parameters per component:")
    for name, count in sorted(model.param_count_by_layer().items()):
        print(f"  {name}: {count:,}")

    batch_size = 8
    seq_len = 64
    epochs = 30

    batches = get_batches(data, batch_size, seq_len)
    print(f"\nBatches per epoch: {len(batches)}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")

    param_list = list(model.params())
    optimizer = SGDMomentum(model, lr=0.01, momentum=0.9)
    optimizer.set_params(param_list)

    print(f"\n{'='*60}")
    print(f"Training started: {epochs} epochs, {len(batches)} batches/epoch")
    print(f"{'='*60}")

    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (xb, yb) in enumerate(batches):
            logits = model.forward(xb)
            loss, dlogits = cross_entropy_loss(logits, yb)

            model.backward(dlogits)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss

        avg_loss = epoch_loss / len(batches)
        perplexity = np.exp(avg_loss)
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | Time: {elapsed:.2f}s")

        if (epoch + 1) % 5 == 0:
            context = "O Romeo, Romeo! wherefore art thou"
            context_ids = np.array([[char_to_idx.get(c, 0) for c in context]])
            output_ids = model.generate(
                context_ids,
                max_new_tokens=50,
                temperature=0.7,
                top_k=10
            )
            generated = ''.join([idx_to_char[i] for i in output_ids[0]])
            print(f"\n  Sample [{epoch+1}]: {generated}\n")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    context = "O Romeo, Romeo! wherefore art thou"
    context_ids = np.array([[char_to_idx.get(c, 0) for c in context]])

    print("\n--- Generation with different temperatures (Top-k=10) ---")
    for temp in [0.3, 0.7, 1.0]:
        output_ids = model.generate(
            context_ids,
            max_new_tokens=100,
            temperature=temp,
            top_k=10
        )
        generated = ''.join([idx_to_char[i] for i in output_ids[0]])
        print(f"\nTemperature = {temp}:")
        print(f"  {generated}")

    print("\n--- Generation with different Top-k (Temperature=0.7) ---")
    for k in [5, 20, None]:
        output_ids = model.generate(
            context_ids,
            max_new_tokens=100,
            temperature=0.7,
            top_k=k
        )
        generated = ''.join([idx_to_char[i] for i in output_ids[0]])
        print(f"\nTop-k = {k}:")
        print(f"  {generated}")

    print(f"\nFinal model parameters: {model.count_params():,}")


if __name__ == '__main__':
    train()
