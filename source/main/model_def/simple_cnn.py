import torch
from torch import nn, optim

from utils import pytorch_utils


class SimpleCNN(nn.Module):

    def __init__(self, enc_embedding_weight):
        super(SimpleCNN, self).__init__()
        self.input_embedding = create_my_embedding(enc_embedding_weight)

        # Conv1d slides on last axis
        self.conv1 = nn.Conv1d(in_channels=300, out_channels=512, kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=7)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=7)
        self.conv3_bn = nn.BatchNorm1d(2048)
        self.max_pool = nn.MaxPool1d(kernel_size=7, stride=1)

        self.dropout = nn.Dropout(p=0.2)

        self.fc = nn.Linear(in_features=2048, out_features=2)

        self.relu = nn.ReLU()
        self.xent = None
        self.optimizer = None
        self.loss_class_weight = torch.tensor([1., 1.])

    def inner_forward(self, input_word, *params):
        """

        :param input_word shape == (batch_size, max_word_len)
        :return: Tensor shape == (batch, max_len, vocab_size)
        """

        # shape == (batch_size, max_word_len, hidden_size)
        word_embed = self.input_embedding(input_word)

        # shape == (batch_size, hidden_size, max_word_len)
        word_embed_permuted = word_embed.permute(0, 2, 1)

        word_pipe = self.conv1(word_embed_permuted)
        word_pipe = self.relu(word_pipe)

        word_pipe = self.conv2(word_pipe)
        word_pipe = self.relu(word_pipe)

        word_pipe = self.conv3(word_pipe)
        word_pipe = self.relu(word_pipe)
        word_pipe = self.conv3_bn(word_pipe)

        word_pipe = self.max_pool(word_pipe)

        # shape == (batch_size, _, _)
        word_pipe = word_pipe.permute(0, 2, 1)

        # shape == (batch_size, max_word_len, no_classes)
        output = self.fc(word_pipe)
        output = self.relu(output)
        output = self.dropout(output)

        return output

    def train(self, mode=True):
        if self.xent is None:
            # Never use `mean`, it does not care about my weight
            self.xent = nn.CrossEntropyLoss(reduction='none', weight=self.loss_class_weight)
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        super().train(mode)

    def get_loss(self, word_input, target, length):
        """

        :param word_input: shape == (batch_size, max_len)
        :param target: shape == (batch_size, max_len)
        :param length: shape == (batch_size)
        :return:
        """
        max_length = word_input.size(1)

        # shape == (batch, max_len, vocab_size)
        predict = self.inner_forward(word_input)
        # shape == (batch, vocab_size, max_len)
        predict = predict.permute(0, 2, 1)

        loss = self.xent(predict, target)
        loss_mask = pytorch_utils.length_to_mask(length, max_len=max_length, dtype=torch.float)
        loss = torch.mul(loss, loss_mask)
        loss = torch.div(loss.sum(dim=1), length.float())
        loss = loss.mean(dim=0)
        return loss

    def forward(self, input_word, *params):
        """

        :param input_word: shape == (batch, max_len)
        :param params:
        :return: Tensor shape == (batch, max_len, vocab_size)
        """

        logits = self.inner_forward(input_word)
        return torch.argmax(logits, dim=2)



    def train_batch(self, word_input, target, length):
        """

        :param word_input: shape == (batch_size, max_len)
        :param target: shape == (batch_size, max_len)
        :return:
        """
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(word_input, target, length)
        loss.backward()
        self.optimizer.step()

        return loss.item()


def create_my_embedding(weights):
    """

    :param weights: numpy array
    :return:
    """
    embedding = nn.Embedding.from_pretrained(torch.from_numpy(weights).float())
    return embedding
