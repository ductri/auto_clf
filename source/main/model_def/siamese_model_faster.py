import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F


class SiameseModel(nn.Module):
    MARGIN = 10

    def __init__(self, core_model):
        super(SiameseModel, self).__init__()
        self.core_model = core_model

        self.my_softmax = nn.Softmax(dim=1)

    def _get_inner_repr(self, input_word):
        """

        :param input_word: shape == (batch_size, max_word_len)
        :return:
        """
        logits = self.core_model(input_word)
        return logits

    def forward(self, i_p, i_n, *args):
        """

        :param i_a: shape == (batch_size, max_word_len)
        :param i_p: shape == (batch_size, max_word_len)
        :param i_n: shape == (batch_size, max_word_len)
        :param args:
        :return:
        """
        i_p_repr = self._get_inner_repr(i_p)
        i_n_repr = self._get_inner_repr(i_n)

        return i_p_repr, i_n_repr

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
    def triplet_loss(vec_p, vec_n):
        """

        :param vec_p: shape == (batch_size, hidden_size)
        :param vec_n: shape == (batch_size, hidden_size)
        :return:
        """
        batch_size = vec_p.size(0)
        random_indices = torch.randperm(batch_size)
        vec_a = vec_p[random_indices]
        loss = SiameseModel.my_distance(vec_a, vec_p) - SiameseModel.my_distance(vec_a, vec_n)+ SiameseModel.MARGIN
        loss = F.relu(loss)
        loss = torch.mean(loss, dim=0)
        return loss

    def build_stuff_for_training(self, device):
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def train_batch(self, i_p, i_n):
        self.train()
        self.optimizer.zero_grad()

        o_p, o_n = self.forward(i_p, i_n)
        loss = SiameseModel.triplet_loss(o_p, o_n)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_distance(self, i1, i2):
        o1 = self._get_inner_repr(i1)
        o2 = self._get_inner_repr(i2)
        return SiameseModel.my_distance(o1, o2)

    def get_distance_anchors(self, anchors, docs):

        anchors_vec = self._get_inner_repr(anchors)
        vec_size = anchors_vec.size(1)
        anchors_vec = anchors_vec.repeat(1, len(docs)).view(-1, vec_size)

        docs_vec = self._get_inner_repr(docs)
        docs_vec = docs_vec.repeat(len(anchors), 1)

        # batch*num_anchors
        dis = SiameseModel.my_distance(anchors_vec, docs_vec)
        batch_size = len(anchors)
        return dis.view(batch_size, -1)
