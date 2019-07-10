from torch import nn, optim
from torch.nn import functional as F


class WrapSiameseModelCore(nn.Module):
    MARGIN = 10

    def __init__(self, core_model):
        super(WrapSiameseModelCore, self).__init__()
        self.core_model = core_model
        self.output_mapping = nn.Linear(10, 2)
        self.xent = None
        self.optimizer = None

    def inner_forward(self, input_word):
        """

        :param input_word: shape == (batch_size, max_word_len)
        :return:
        """
        logits = self.core_model(input_word)
        logits = F.relu(logits)
        logits = self.output_mapping(logits)
        return logits

    def forward(self, input_word, *input):
        logits = self.inner_forward(input_word)
        return F.softmax(logits, dim=1)

    def train(self, mode=True):
        if self.xent is None:
            # Never use `mean`, it does not care about my weight
            self.xent = nn.CrossEntropyLoss(reduction='none')
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        super().train(mode)

    def get_loss(self, word_input, target):
        """

        :param word_input: shape == (batch, seq_len)
        :param target: shape == (batch)
        :return:
        """
        logits = self.inner_forward(word_input)
        loss = self.xent(logits, target)
        loss = loss.mean(dim=0)
        return loss

    def train_batch(self, word_input, target):
        self.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(word_input, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


