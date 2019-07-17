import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class SiameseModelCore(nn.Module):
    MARGIN = 10

    def __init__(self, embedding_weight):
        super(SiameseModelCore, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=300, num_layers=3, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=1200, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

        if embedding_weight is not None:
            self.input_embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weight).float(), freeze=False)
        else:
            self.input_embedding = nn.Embedding(num_embeddings=20002,embedding_dim=300)

        self.my_softmax = nn.Softmax(dim=1)

    def forward(self, input_word):
        """

        :param input_word: shape == (batch_size, max_word_len)
        :return:
        """
        word_embed = self.input_embedding(input_word)

        # shape == (max_word_len, batch_size, hidden_size)
        word_embed_permuted = word_embed.permute(1, 0, 2)

        # h_n, c_n each has shape == (2*2, batch_size, hidden_size)
        # outputs shape == (max_len, batch_size, 2*hidden_size)
        _, (h_n, c_n) = self.lstm(word_embed_permuted)

        batch_size = h_n.size(1)
        hidden_size = h_n.size(2)

        num_layer = 3
        num_direction = 2

        h_n = h_n.view(num_layer, num_direction, batch_size, hidden_size)
        h_n = h_n[1]
        h_n = h_n.permute(1, 0, 2).contiguous()
        # shape == (batch_size, 2*hidden_size)
        h_n = h_n.view(batch_size, -1)

        c_n = c_n.view(num_layer, num_direction, batch_size, hidden_size)
        c_n = c_n[1]
        c_n = c_n.permute(1, 0, 2).contiguous()
        # shape == (batch_size, 2*hidden_size)
        c_n = c_n.view(batch_size, -1)

        # shape == (batch_size, 2*hidden_size)
        final_state = torch.cat((h_n, c_n), dim=1)

        # shape == (batch_size, 2*2*hidden_size)
        word_pipe = self.dropout(final_state)

        logits = self.fc1(word_pipe)
        logits = F.relu(logits)
        logits = self.fc2(logits)

        return logits
