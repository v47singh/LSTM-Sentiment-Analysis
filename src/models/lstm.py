import torch
from torch import nn, optim
import torch.nn.functional as F


class LSTMClassifier(nn.Module):
    """docstring for LSTMClassifier"""
    def __init__(self, config, embedding_wts, n_lables):
        super(LSTMClassifier, self).__init__()
        self.config = config
        self.dropout = config['dropout']
        self.embedding_wts = embedding_wts
        self.output_dim = n_lables
        self.embedding = nn.Embedding.from_pretrained(embedding_wts,
                                                      freeze=False)
        self.embedding_dropout = nn.Dropout(config['dropout'])
        self.mse_loss = nn.MSELoss()
        self.rnn = nn.LSTM(self.config['embedding_dim'],
                           self.config['hidden_dim'],
                           num_layers=self.config['n_layers'],
                           dropout=self.dropout,
                           bidirectional=True,
                           bias=True)
        if self.config['operation'] == 'cat':
            self.out = nn.Linear(2*self.config['hidden_dim'], self.output_dim)
        else:
            self.out = nn.Linear(self.config['hidden_dim'], self.output_dim)

        if self.config['operation'] == 'transfuse':
            self.transfuse = nn.Sequential(
                nn.Linear(2*self.config['hidden_dim'], self.config['hidden_dim']),
                nn.ReLU()
            )
        elif self.config['operation'] == 'pool1d':
            self.pool1d = F.max_pool1d
            self.out = nn.Linear(4*self.config['hidden_dim'], self.output_dim)

        self.softmax = F.softmax
        self.use_attn = config['use_attn?']

    def aux_loss(self, concat, result):
        """
        Calculates MSE between fw_bw, [tr_hidden; zeros]
        concat (bs x 2*hidden_dim):
            Concatenated forward and backward hidden states
        result (bs x hidden):
            Transfused hidden vector
        """
        if self.config['operation'] not in {'cat', 'pool1d'}:
            return self.mse_loss(concat, torch.cat(
                    (result, torch.zeros(result.size()).to(
                        self.config['device'])), dim=-1))

    def attn(self, rnn_output, final_hidden):
        """
        Returns `attended` hidden state given an rnn_output and its final
        hidden state

        attn: torch.Tensor, torch.Tensor -> torch.Tensor
        requires:
            rnn_output.shape => batch_size x max_seq_len x hidden_dim
            final_hidden.shape => batch_size x hidden_dim
        """
        attn_wts = torch.bmm(rnn_output, final_hidden.unsqueeze(2)).squeeze(2)
        soft_attn_wts = self.softmax(attn_wts, dim=1)  # bs x num_t
        # In the next step, rnn_output.shape changes as follows:
        # (bs x num_t x hidden_dim) => (bs x hidden_dim x num_t)
        # Finally, we get new_hidden of shape: (bs x hidden_dim)
        new_hidden = torch.bmm(rnn_output.transpose(1, 2),
                               soft_attn_wts.unsqueeze(2)).squeeze(2)
        return new_hidden

    def fuse(self, logits):
        fwd = logits[0, :, :self.config['hidden_dim']]
        bwd = logits[1, :, :self.config['hidden_dim']]

        # t x bs x 2*hidden
        concat = torch.cat((fwd, bwd), dim=-1)
        if self.config['operation'] == 'sum':
            result = fwd + bwd
        elif self.config['operation'] == 'average':
            result = (fwd + bwd)/2
        elif self.config['operation'] == 'cat':
            result = concat
        elif self.config['operation'] == 'transfuse':
            result = self.transfuse(concat)
        elif self.config['operation'] == 'pool1d':
            logits = logits.permute(1, 2, 0).contiguous()
            result = self.pool1d(logits, logits.size(2)).squeeze()

        return result, self.aux_loss(concat, result)

    def forward(self, input_seq_batch, input_lengths):
        max_seq_length, bs = input_seq_batch.size()
        embedded = self.embedding_dropout(self.embedding(input_seq_batch))

        rnn_output, (hidden, _) = self.rnn(embedded)

        if self.use_attn:
            # Do this to sum the bidirectional outputs
            rnn_output = rnn_output.view(2, max_seq_length, bs,
                                         self.config['hidden_dim'])

            rnn_output, aux_loss_1 = self.fuse(rnn_output)
            rnn_output = rnn_output.permute(1, 0, 2)  # bs x num_t x hidden_dim

            final_hidden = hidden.view(self.config['n_layers'], 2,
                                       bs, self.config['hidden_dim'])[-1]
            # sum (pool) forward and backward hidden states
            final_hidden, aux_loss_2 = self.fuse(final_hidden)

            attended_ouptut = self.attn(rnn_output, final_hidden)
            aux_loss = (aux_loss_1 + aux_loss_2) if aux_loss_1 else 0
            return self.out(attended_ouptut), aux_loss

        # Here, final_hidden.shape becomes => (bs x hidden_dim)
        final_hidden = hidden.view(self.config['n_layers'], 2,
                                   bs, self.config['hidden_dim'])[-1]

        if self.config['operation'] == 'pool1d':
            rnn_output = rnn_output.permute(1, 2, 0)
            rnn_output = self.pool1d(rnn_output, rnn_output.size(2)).squeeze()
            # bs x 2*hidden
            rnn_output = rnn_output.view(bs, 2, self.config['hidden_dim'])
            # bs x hidden
            rnn_output = torch.cat((rnn_output[:, 0, :], rnn_output[:, 1, :]), dim=-1)
            final_hidden = torch.cat(
                (final_hidden[0, :, :self.config['hidden_dim']],
                    final_hidden[1, :, :self.config['hidden_dim']]),
                dim=-1)
            # bs x 4*hidden
            concat = torch.cat((final_hidden, rnn_output), dim=-1)
            return self.out(concat), 0

        final_hidden, aux_loss = self.fuse(final_hidden)
        return self.out(final_hidden), (aux_loss if aux_loss else 0)
