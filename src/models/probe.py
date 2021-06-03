"""Classes for specifying probe pytorch modules."""
from abc import ABC

import torch.nn as nn
import torch


class Probe(nn.Module, ABC):
    pass


class TwoWordPSDProbe(Probe):
    """ Computes squared L2 distance after projection by a probe.

  For a batch of sentences, computes all n^2 pairs of distances
  for each sentence in the batch.
  """

    def __init__(self, args):
        print('Constructing TwoWordPSDProbe')
        super(TwoWordPSDProbe, self).__init__()
        self.args = args
        self.probe_rank = args['probe']['maximum_rank']
        self.num_layers = args['probe'].get('num_layers')
        if self.num_layers is None:
            print("Number of probe layers unspecified; defaulting to 1")
            self.num_layers = 1
        self.model_dim = args['model']['hidden_dim']
        last_embedding_dim = self.model_dim
        hidden_dim = 1024
        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers - 1):
            self.layers.append(nn.Linear(last_embedding_dim, hidden_dim))
            last_embedding_dim = hidden_dim
        self.layers.append(nn.Linear(last_embedding_dim, self.probe_rank))
        self.to(args['device'])

    def forward(self, batch):
        """ Computes all n^2 pairs of distances after projection
    for each sentence in a batch.

    Note that due to padding, some distances will be non-zero for pads.
    Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
    """
        transformed = batch
        for layer in self.layers:
            transformed = layer(transformed).clamp(min=0)
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = torch.sum(squared_diffs, -1)
        return squared_distances


class OneWordPSDProbe(Probe):
    def __init__(self, args):
        print('Constructing OneWordPSDProbe')
        super(OneWordPSDProbe, self).__init__()
        self.args = args
        self.probe_rank = args['probe']['maximum_rank']
        self.num_layers = args['probe'].get('num_layers')
        if self.num_layers is None:
            print("Number of probe layers unspecified; defaulting to 1")
            self.num_layers = 1
        self.model_dim = args['model']['hidden_dim']
        last_embedding_dim = self.model_dim
        hidden_dim = 1024
        self.layers = nn.ModuleList()
        for layer_idx in range(self.num_layers - 1):
            self.layers.append(nn.Linear(last_embedding_dim, hidden_dim))
            last_embedding_dim = hidden_dim
        self.layers.append(nn.Linear(last_embedding_dim, self.probe_rank))
        self.to(args['device'])

    def forward(self, batch):
        """ Computes all n depths after projection
    for each sentence in a batch.

    Args:
      batch: a batch of word representations of the shape
        (batch_size, max_seq_len, representation_dim)
    Returns:
      A tensor of depths of shape (batch_size, max_seq_len)
    """
        transformed = batch
        for layer in self.layers:
            transformed = layer(transformed).clamp(min=0)
        batchlen, seqlen, rank = transformed.size()
        norms = torch.bmm(transformed.view(batchlen * seqlen, 1, rank),
                          transformed.view(batchlen * seqlen, rank, 1))
        norms = norms.view(batchlen, seqlen)
        return norms
