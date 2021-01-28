import torch
from pytorch_metric_learning.losses import BaseMetricLossFunction
from .util import get_pairwise_stds, get_pairwise_similarity, dist

class RSTopKPreLoss(BaseMetricLossFunction):
    """
    Deep Metric Learning Based on Rank-sensitive Optimization of Top-k Precision
    Naoki Muramoto and Hai-Tao Yu. In Proceedings of CIKM’20.
    """

    def __init__(self, k=4):
        super().__init__()

        self.name = 'RSTopKPreLoss'
        self.k = k
        self.margin = 0.1

    def compute_loss(self, embeddings, labels, indices_tuple):
        '''
        assuming no-existence of classes with a single instance == samples_per_class > 1
        :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
        :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
        :param k: cutoff value
        :param margin:
        :return:
        '''

        simi_mat = get_pairwise_similarity(batch_reprs=embeddings)
        cls_match_mat = get_pairwise_stds(
            batch_labels=labels)

        simi_mat_hat = simi_mat + (1.0 - cls_match_mat) * self.margin  # impose margin

        ''' get rank positions '''
        _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
        _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
        rank_mat = desc_indice + 1.  # todo using desc_indice directly without (+1) to improve efficiency

        batch_pos_nums = torch.sum(cls_match_mat, dim=1)

        ''' get proper K rather than a rigid predefined K
        torch.clamp(tensor, min=value) is cmax and torch.clamp(tensor, max=value) is cmin.
        It works but is a little confusing at first.
        '''
        # batch_ks = torch.clamp(batch_pos_nums, max=k)
        '''
        due to no explicit self-similarity filtering.
        implicit assumption: a common L2-normalization leading to self-similarity of the maximum one!
        '''
        batch_ks = torch.clamp(batch_pos_nums, max=self.k + 1)
        k_mat = batch_ks.view(-1, 1).repeat(1, rank_mat.size(1))

        '''
        Only deal with a single case: n_{+}>=k
        step-1: determine set of false positive neighbors, i.e., N, i.e., cls_match_std is zero && rank<=k

        step-2: determine the size of N, i.e., |N| which determines the size of P

        step-3: determine set of false negative neighbors, i.e., P, i.e., cls_match_std is one && rank>k && rank<= (k+|N|)
        '''
        # N
        batch_false_pos = (cls_match_mat < 1) & (rank_mat <= k_mat)  # torch.uint8 -> used as indice
        batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1)  # used as one/zero

        # P
        batch_false_negs = cls_match_mat.bool() & (rank_mat > k_mat)  # all false negative

        ''' just for check '''
        # batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
        # print('batch_fn_nums', batch_fn_nums)

        batch_loss = torch.tensor(0., requires_grad=True).cuda()
        for i in range(cls_match_mat.size(0)):
            fp_num = int(batch_fp_nums.data[i].item())
            if fp_num > 0:  # error exists, in other words, skip correct case
                all_false_neg = simi_mat_hat[i, :][batch_false_negs[i, :]]
                rank_neg = rank_mat[i, :][batch_false_negs[i, :]]
                top_false_neg, neg_idx = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
                rank_top_neg = torch.gather(rank_neg, -1, neg_idx)
                ks = torch.zeros(fp_num).cuda()
                batch_ones = torch.ones_like(ks)
                ks0 = ks.add(self.k)
                # ks1 = ks.add(k+1)
                ks3 = ks.add(self.k + 3)
                beta1 = torch.add(batch_ones, -1 / (rank_top_neg - ks0))
                loss_neg = 3 * torch.dot(beta1, top_false_neg) / fp_num

                rank_pos = rank_mat[i, :][batch_false_pos[i, :]]
                false_pos = simi_mat_hat[i, :][batch_false_pos[i, :]]
                beta2 = torch.add(batch_ones, -1 / (ks3 - rank_pos))
                loss_pos = 3 * torch.dot(beta2, false_pos) / fp_num

                loss = torch.sum(loss_pos - loss_neg)
                batch_loss += loss

        return {
            "loss": {
                "losses": batch_loss,
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }