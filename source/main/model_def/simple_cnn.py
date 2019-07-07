import torch
from torch import nn, optim
from torch.nn import functional as F

from utils import pytorch_utils


class SimpleCNN(nn.Module):

    def __init__(self, enc_embedding_weight):
        super(SimpleCNN, self).__init__()
        self.input_embedding = create_my_embedding(enc_embedding_weight)

        # Conv1d slides on last axis
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=300, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=300, out_channels=128, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)
        self.conv4 = nn.Conv1d(in_channels=128*3, out_channels=128*5, kernel_size=7)
        self.conv5 = nn.Conv1d(in_channels=128*5, out_channels=128*7, kernel_size=7)

        self.dropout = nn.Dropout(p=0.0)

        self.fc = nn.Linear(in_features=32256, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)

        self.relu = nn.ReLU()
        # self.xent = None
        # self.optimizer = None
        # self.loss_class_weight = torch.tensor([1., 1.])

    def inner_forward(self, input_word, *params):
        """

        :param input_word shape == (batch_size, max_word_len)
        :return: Tensor shape == (batch, 2)
        """

        # shape == (batch_size, max_word_len, hidden_size)
        word_embed = self.input_embedding(input_word)

        # shape == (batch_size, hidden_size, max_word_len)
        word_embed_permuted = word_embed.permute(0, 2, 1)

        word_pipe = self.conv1(self.dropout(word_embed_permuted))
        word_pipe1 = self.relu(word_pipe)

        word_pipe = self.conv2(self.dropout(word_embed_permuted))
        word_pipe2 = self.relu(word_pipe)

        word_pipe = self.conv3(self.dropout(word_embed_permuted))
        word_pipe3 = self.relu(word_pipe)

        word_pipe = torch.cat((word_pipe1, word_pipe2, word_pipe3), dim=1)
        word_pipe = self.pool(word_pipe)
        word_pipe = self.conv4(word_pipe)
        word_pipe = self.conv5(word_pipe)

        batch_size = input_word.size(0)
        word_pipe = word_pipe.view(batch_size, -1)
        word_pipe = F.relu(word_pipe)
        output = self.fc(word_pipe)
        output = self.fc2(output)
        # output = torch.where(output >= 1, torch.ones_like(output), output)
        # output = F.relu(output)
        return output

    # def train(self, mode=True):
    #     if self.xent is None:
    #         # Never use `mean`, it does not care about my weight
    #         self.xent = nn.CrossEntropyLoss(reduction='none', weight=self.loss_class_weight)
    #     if self.optimizer is None:
    #         # self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
    #         self.optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.1)
    #     super().train(mode)

    # def get_loss(self, word_input, target):
    #     """
    #
    #     :param word_input: shape == (batch_size, max_len)
    #     :param target: shape == (batch_size, max_len)
    #     :param length: shape == (batch_size)
    #     :return:
    #     """
    #     # shape == (batch, 2)
    #     predict = self.inner_forward(word_input)
    #     # shape == (batch, 2)
    #     loss = self.xent(predict, target)
    #     loss = loss.mean(dim=0)
    #     return loss

    # def forward(self, input_word, *params):
    #     """
    #
    #     :param input_word: shape == (batch, max_len)
    #     :param params:
    #     :return: Tensor shape == (batch, max_len, vocab_size)
    #     """
    #
    #     logits = self.inner_forward(input_word)
    #     return F.softmax(logits, dim=1)

    # def train_batch(self, word_input, target):
    #     """
    #
    #     :param word_input: shape == (batch_size, max_len)
    #     :param target: shape == (batch_size, max_len)
    #     :return:
    #     """
    #     self.train()
    #     self.optimizer.zero_grad()
    #     loss = self.get_loss(word_input, target)
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     return loss.item()


def create_my_embedding(weights):
    """

    :param weights: numpy array
    :return:
    """
    embedding = nn.Embedding.from_pretrained(torch.from_numpy(weights).float())
    return embedding
