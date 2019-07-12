import torch
from torch import nn, optim
from torch.nn import functional as F


class SiameseModel(nn.Module):
    MARGIN = 1

    def __init__(self, core_model):
        super(SiameseModel, self).__init__()
        self.core_model = core_model

    def forward(self, input_word, *args):
        """

        :param i_a: shape == (batch_size, max_word_len)
        :param i_p: shape == (batch_size, max_word_len)
        :param i_n: shape == (batch_size, max_word_len)
        :param args:
        :return:
        """
        inner_repr = self.core_model(input_word)
        return inner_repr

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
    def triplet_loss(o_p_o_n, positive_size):
        """

        :param vec_a: shape == (batch_size, hidden_size)
        :param vec_p: shape == (batch_size, hidden_size)
        :param vec_n: shape == (batch_size, hidden_size)
        :return:
        """
        o_a = torch.cat((o_p_o_n[1:positive_size], o_p_o_n[0:1]), dim=0)
        o_p = o_p_o_n[:positive_size]
        o_n = o_p_o_n[positive_size:]
        loss = SiameseModel.my_distance(o_a, o_p) - SiameseModel.my_distance(o_a, o_n) + SiameseModel.MARGIN
        loss = F.relu(loss)
        loss = torch.mean(loss, dim=0)
        return loss

    def build_stuff_for_training(self, device):
        # self.optimizer = optim.SGD(self.parameters(), lr=0.1)
        self.optimizer = optim.Adam(self.parameters(), lr=4e-3)

    def get_loss(self, i_p, i_n):
        positive_size = i_p.size(0)
        i_p_i_n = torch.cat((i_p, i_n), dim=0)
        o_p_o_n = self.forward(i_p_i_n)
        loss = SiameseModel.triplet_loss(o_p_o_n, positive_size)
        return loss

    def train_batch(self, i_p, i_n):
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(i_p, i_n)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_distance(self, i1, i2):
        o1 = self._get_inner_repr(i1)
        o2 = self._get_inner_repr(i2)
        return SiameseModel.my_distance(o1, o2)

    def get_distance_anchors(self, anchors, docs):
        anchors_vec = self.forward(anchors)
        vec_size = anchors_vec.size(1)
        anchors_vec = anchors_vec.repeat(1, len(docs)).view(-1, vec_size)

        docs_vec = self.forward(docs)
        docs_vec = docs_vec.repeat(len(anchors), 1)

        # batch*num_anchors
        dis = SiameseModel.my_distance(anchors_vec, docs_vec)
        batch_size = len(anchors)
        return dis.view(batch_size, -1)
