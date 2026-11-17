import numpy as np
from segtok import tokenizer
import torch as th
from torch import nn

# Using a basic RNN/LSTM for Language modeling
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, rnn_size, num_layers=1, dropout=0, tie_weights=False):
        super().__init__()
        
        # Embedding layer of shape [vocab_size, rnn_size]
        self.embedding = nn.Embedding(vocab_size, rnn_size)

        # LSTM layer (batch_first convention)
        self.lstm = nn.LSTM(input_size=rnn_size, hidden_size=rnn_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        # Dropout on the embeddings and on the last hidden output
        self.dropout = nn.Dropout(dropout)
        # Dense layer projecting LSTM outputs to vocabulary logits
        self.output = nn.Linear(rnn_size, vocab_size)
        # Optional weight tying: output layer shares the embedding matrix
        if tie_weights:
            self.output.weight = self.embedding.weight
        
    def forward(self, x):
        embeds = self.dropout(self.embedding(x))   # [batch, seq, rnn_size]
        lstm_out, _ = self.lstm(embeds)            # [batch, seq, rnn_size]
        lstm_drop = self.dropout(lstm_out)
        logits = self.output(lstm_drop)            # [batch, seq, vocab_size]
        return logits
