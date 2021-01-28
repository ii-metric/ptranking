import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from .util import get_pairwise_stds, get_pairwise_similarity, dist
import torch.nn.functional as F

class ListNet(BaseMetricLossFunction):

    def __init__(self): # , anchor_id='Anchor', use_similarity=False, opt=None
        super(ListNet, self).__init__()

        self.name = 'listnet'

    def forward(self, embeddings, labels, indices_tuple):
        '''
        :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
        :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
        :return:
        '''

        cls_match_mat = get_pairwise_stds(batch_labels=labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise

        # if self.use_similarity:
        #     sim_mat = get_pairwise_similarity(batch_reprs=batch)
        # else:
        dist_mat = dist(batch_reprs=embeddings, squared=False)  # [batch_size, batch_size], pairwise distances
        sim_mat = -dist_mat

        # convert to one-dimension vector
        batch_size = embeddings.size(0)
        index_mat = torch.triu(torch.ones(batch_size, batch_size), diagonal=1) == 1
        sim_vec = sim_mat[index_mat]
        cls_vec = cls_match_mat[index_mat]

        # cross-entropy between two softmaxed vectors
        # batch_loss = -torch.sum(F.softmax(sim_vec) * F.log_softmax(cls_vec))
        batch_loss = -torch.sum(F.softmax(cls_vec) * F.log_softmax(sim_vec))

        return batch_loss
