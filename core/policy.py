import torch
from torch.nn import Linear, LSTMCell, Module, Embedding
from torch.nn.init import uniform_
import torch.nn.functional as F
import numpy as np

device = 'cpu'


class CriticNet(Module):
    def __init__(self, hidden_size):
        super(CriticNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, 1)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = torch.tanh(self.l2(x))
        return x


class ActorNet(Module):
    def __init__(self, hidden_size, num_programs):
        super(ActorNet, self).__init__()
        self.l1 = Linear(hidden_size, hidden_size//2)
        self.l2 = Linear(hidden_size//2, num_programs)

    def forward(self, hidden_state):
        x = F.relu(self.l1(hidden_state))
        x = F.softmax(self.l2(x), dim=-1)
        return x


class Policy(Module):
    """This class represents the NPI policy containing the environment encoder, the key-value and program embedding
    matrices, the NPI core lstm and the value networks for each task.

    Args:
        encoder (:obj:`{HanoiEnvEncoder, ListEnvEncoder, RecursiveListEnvEncoder, PyramidsEnvEncoder}`):
        hidden_size (int): Dimensionality of the LSTM hidden state
        num_programs (int): Overall number of programs and size actor's output softmax vector
        num_non_primary_programs (int): Number of non-zero level programs, also number of rows in embedding matrix
        embedding_dim (int): Dimensionality of the programs' embedding vectors
        encoding_dim (int): Dimensionality of the environment observation's encoding
        indices_non_primary_programs (list): Non zero level programs' indices
        learning_rate (float, optional): Defaults to 10^-3.
    """
    def __init__(self, encoder, hidden_size, num_programs, num_non_primary_programs, embedding_dim,
                 encoding_dim, indices_non_primary_programs, learning_rate=1e-3):

        super(Policy, self).__init__()

        self._uniform_init = (-0.1, 0.1)

        self._hidden_size = hidden_size
        self.num_programs = num_programs
        self.num_non_primary_programs = num_non_primary_programs

        self.embedding_dim = embedding_dim
        self.encoding_dim = encoding_dim

        # Initialize networks
        self.Mprog = Embedding(num_non_primary_programs, embedding_dim)
        self.encoder = encoder

        self.lstm = LSTMCell(self.encoding_dim + self.embedding_dim, self._hidden_size)
        self.critic = CriticNet(self._hidden_size)
        self.actor = ActorNet(self._hidden_size, self.num_programs)

        self.init_networks()
        self.init_optimizer(lr=learning_rate)

        # Compute relative indices of non primary programs (to deal with task indices)
        self.relative_indices = dict((prog_idx, relat_idx) for relat_idx, prog_idx in enumerate(indices_non_primary_programs))

    def init_networks(self):

        for p in self.encoder.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.lstm.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.critic.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

        for p in self.actor.parameters():
            uniform_(p, self._uniform_init[0], self._uniform_init[1])

    def init_optimizer(self, lr):
        '''Initialize the optimizer.

        Args:
            lr (float): learning rate
        '''
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _one_hot_encode(self, digits, basis=6):
        """One hot encode a digit with basis. The digit may be None,
        the encoding associated to None is a vector full of zeros.

        Args:
          digits: batch (list) of digits
          basis:  (Default value = 6)

        Returns:
          a numpy array representing the 10-hot-encoding of the digit

        """
        encoding = torch.zeros(len(digits), basis)
        digits_filtered = list(filter(lambda x: x is not None, digits))

        if len(digits_filtered) != 0:
            tmp = [[idx for idx, digit in enumerate(digits) if digit is not None], digits_filtered]
            encoding[tmp] = 1.0
        return encoding

    def predict_on_batch(self, e_t, i_t, h_t, c_t):
        """Run one NPI inference.

        Args:
          e_t: batch of environment observation
          i_t: batch of calling program
          h_t: batch of lstm hidden state
          c_t: batch of lstm cell state

        Returns:
          probabilities over programs, value, new hidden state, new cell state

        """
        batch_size = len(i_t)
        s_t = self.encoder(e_t.view(batch_size, -1))
        relative_prog_indices = [self.relative_indices[idx] for idx in i_t]
        p_t = self.Mprog(torch.LongTensor(relative_prog_indices)).view(batch_size, -1)

        new_h, new_c = self.lstm(torch.cat([s_t, p_t], -1), (h_t, c_t))

        actor_out = self.actor(new_h)
        critic_out = self.critic(new_h)
        return actor_out, critic_out, new_h, new_c

    def train_on_batch(self, batch):
        """perform optimization step.

        Args:
          batch (tuple): tuple of batches of environment observations, calling programs, lstm's hidden and cell states

        Returns:
          policy loss, value loss, total loss combining policy and value losses
        """
        e_t = torch.FloatTensor(np.stack(batch[0]))
        i_t = batch[1]
        lstm_states = batch[2]
        h_t, c_t = zip(*lstm_states)
        h_t, c_t = torch.squeeze(torch.stack(list(h_t))), torch.squeeze(torch.stack(list(c_t)))

        policy_labels = torch.squeeze(torch.stack(batch[3]))
        value_labels = torch.stack(batch[4]).view(-1, 1)

        self.optimizer.zero_grad()
        policy_predictions, value_predictions, _, _ = self.predict_on_batch(e_t, i_t, h_t, c_t)

        policy_loss = -torch.mean(policy_labels * torch.log(policy_predictions), dim=-1).mean()
        value_loss = torch.pow(value_predictions - value_labels, 2).mean()

        total_loss = (policy_loss + value_loss) / 2
        total_loss.backward()
        self.optimizer.step()

        return policy_loss, value_loss, total_loss

    def forward_once(self, e_t, i_t, h, c):
        """Run one NPI inference using predict.

        Args:
          e_t: current environment observation
          i_t: current program calling
          h: previous lstm hidden state
          c: previous lstm cell state

        Returns:
          probabilities over programs, value, new hidden state, new cell state, a program index sampled according to
          the probabilities over programs)

        """
        e_t = torch.FloatTensor(e_t)
        e_t, h, c = e_t.view(1, -1), h.view(1, -1), c.view(1, -1)
        with torch.no_grad():
            e_t = e_t.to(device)
            actor_out, critic_out, new_h, new_c = self.predict_on_batch(e_t, [i_t], h, c)
        return actor_out, critic_out, new_h, new_c

    def init_tensors(self):
        """Creates tensors representing the internal states of the lstm filled with zeros.
        
        Returns:
            instantiated hidden and cell states
        """
        h = torch.zeros(1, self._hidden_size)
        c = torch.zeros(1, self._hidden_size)
        h, c = h.to(device), c.to(device)
        return h, c

