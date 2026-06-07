import torch
import torch.nn as nn

class LSTMLanguageModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_size,
                 hidden_size):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        output, _ = self.lstm(x)

        output = self.fc(output)

        return output