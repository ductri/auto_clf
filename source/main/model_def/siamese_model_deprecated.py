import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class SiameseModel(nn.Module):
    MARGIN = 10

    def __init__(self, embedding_weight):
        super(SiameseModel, self).__init__()
        self.lstm = nn.LSTM(input_size=300, hidden_size=300, num_layers=3, bidirectional=True, dropout=0.5)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=1200, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

        self.input_embedding = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weight).float(), freeze=False)

        self.my_softmax = nn.Softmax(dim=1)

    def _get_inner_repr(self, input_word):
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

    def forward(self, i_a, i_p, i_n, *args):
        """

        :param i_a: shape == (batch_size, max_word_len)
        :param i_p: shape == (batch_size, max_word_len)
        :param i_n: shape == (batch_size, max_word_len)
        :param args:
        :return:
        """
        i_a_repr = self._get_inner_repr(i_a)
        i_p_repr = self._get_inner_repr(i_p)
        i_n_repr = self._get_inner_repr(i_n)

        return i_a_repr, i_p_repr, i_n_repr

    @staticmethod
    def my_distance(vec1, vec2):
        """

        :param vec1: shape == (batch_size, hidden_size)
        :param vec2: shape == (batch_size, hidden_size)
        :return: shape == (batch_size)
        """
        dis = torch.norm(vec1-vec2, 2, dim=1)
        return dis

    @staticmethod
    def triplet_loss(vec_a, vec_p, vec_n):
        """

        :param vec_a: shape == (batch_size, hidden_size)
        :param vec_p: shape == (batch_size, hidden_size)
        :param vec_n: shape == (batch_size, hidden_size)
        :return:
        """
        loss = SiameseModel.my_distance(vec_a, vec_p) - SiameseModel.my_distance(vec_a, vec_n)+ SiameseModel.MARGIN
        loss = F.relu(loss)
        loss = torch.mean(loss, dim=0)
        return loss

    def build_stuff_for_training(self, device):
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def train_batch(self, i_a, i_p, i_n):
        self.train()
        self.optimizer.zero_grad()

        o_a, o_p, o_n = self.forward(i_a, i_p, i_n)
        loss = SiameseModel.triplet_loss(o_a, o_p, o_n)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_distance(self, i1, i2):
        o1 = self._get_inner_repr(i1)
        o2 = self._get_inner_repr(i2)
        return SiameseModel.my_distance(o1, o2)
