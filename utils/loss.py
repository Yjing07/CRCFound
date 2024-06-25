import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Sequence
from torch import Tensor

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        # log_p = F.softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class NegativeLogLikelihood(nn.Module):
    def __init__(self,device):
        super(NegativeLogLikelihood, self).__init__()
        self.device =device

    def forward(self, risk_pred, y, e):
        """
        @params: risk_pred: 预测的生存期/风险函数，即cox回归指数项上的结果，注意该数据与实际生存期间的正负关系（比如风险函数与生存期为法相关系）   shape: (N,1)
        @params: y: 真实事件终止事件（可能为右删失数据，也有可能为真实事件终止）    shape:(N,1)
        @params: e: event indicator， 1-事件终止； 0-右删失     shape:(N,1)
        """
        mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)     # mask矩阵, mask(i,j)中j表示基准事件，i为其它对比事件
        mask[(y.T-y) > 0] = 0             # 基准事件真实存活期大于其它对比事件的，无需考虑
        exp_loss = torch.exp(risk_pred) * mask       # mask非必要项，(N, N)
        log_loss = torch.log((exp_loss.sum(dim=0))/(mask.sum(dim=0)))      # 这里取平均以消除pair中样本长度的影响， (N, 1)
        log_loss = log_loss.reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred - log_loss) * e) / torch.sum(e)       # 不需要计入右删失值
        return neg_log_loss

def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data

    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = torch.relu(hazard_pred).reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox

def cox_loss_torch( score, time_value, event):
        '''
        Args
            score: 		predicted survival time_value, torch tensor of shape (None, )
            time_value:		true survival time_value, torch tensor of shape (None, )
            event:		event, tf tensor of shape (None, )
        Return
            loss:		partial likelihood of cox regression
        '''

        ## cox regression computes the risk score, we want the opposite
        # score   = -score

        ## find index i satisfying event[i]==1
        ix      = torch.where(event>0)[0]  # shape of ix is [None, 1]

        ## sel_mat is a matrix where sel_mat[i,j]==1 where time_value[i]<=time_value[j]
        sel_mat = torch.gather(time_value,0,ix).reshape(-1,1)<=time_value
        
        ## formula: \sum_i[s_i-\log(\sum_j{e^{s_j}})] where time_value[i]<=time_value[j] and event[i]==1
        p_lik   = torch.gather(score,0, ix).reshape(-1,1) \
                        - torch.log(torch.sum(sel_mat * torch.exp(score), axis=-1))
        # p_lik = tf.gather(score, ix) - tf.log(tf.reduce_sum(tf.transpose(tf.exp(score)), axis=-1))
        loss    = -torch.mean(p_lik)

        return loss

class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """
    def __init__(self, alpha=0.0, eps=1e-7, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='mean'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    # print("h shape", h.shape)

    # make sure these are ints
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)
    c = torch.sigmoid(c)
    # print("hazards shape", hazards.shape)

    S = torch.cumprod(1 - hazards, dim=1)
    # print("S.shape", S.shape, S)

    S_padded = torch.cat([torch.ones_like(c), S], 1)
    # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # hazards[y] = hazards(1)
    # S[1] = S(1)
    # TODO: document and check

    # print("S_padded.shape", S_padded.shape, S_padded)


    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y+1).clamp(min=eps)
    # print('s_prev.s_prev', s_prev.shape, s_prev)
    # print('h_this.shape', h_this.shape, h_this)
    # print('s_this.shape', s_this.shape, s_this)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)
    

    # print('uncensored_loss.shape', uncensored_loss.shape)
    # print('censored_loss.shape', censored_loss.shape)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


# def nll_loss(hazards, Y, c, alpha=0.4, eps=1e-7):
#     batch_size = len(Y)
#     Y = Y.view(batch_size, 1).to(torch.int64) # ground truth bin, 1,2,...,k
#     c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    
#     S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
#     # without padding, S(0) = S[0], h(0) = h[0]
#     S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
#     # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
#     #h[y] = h(1)
#     #S[1] = S(1)
#     uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps)))
#     censored_loss = - c * torch.log(torch.gather(S_padded, 1, Y+1).clamp(min=eps))
#     neg_l = censored_loss + uncensored_loss
#     loss = (1-alpha) * neg_l + alpha * uncensored_loss
#     loss = loss.mean()
#     return loss

class kl_loss(nn.Module):
    """
    basic KD loss function based on "Distilling the Knowledge in a Neural Network"
    https://arxiv.org/abs/1503.02531
    :param input_logits: student score map
    :param target_logits: teacher score map
    :param T:  for softmax
    :return: loss value
    """
    def __init__(self,softmax_t=1):
        super(kl_loss, self).__init__()
        self.softmax_t = softmax_t
    def forward(self,input_logits, target_logits,mask = None):
        T = self.softmax_t
        assert input_logits.size() == target_logits.size()
        input_softmax = F.log_softmax(input_logits/T, dim=1)
        target_softmax = F.softmax(target_logits/T, dim=1)
        if mask is None:
            klloss = F.kl_div(input_softmax, target_softmax,reduction='batchmean')
            
            return klloss
        else:
            klloss = (F.kl_div(input_softmax, target_softmax,reduction='none')).mean((1,2,3))*mask
            return klloss.mean()
        
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # 计算样本之间的欧氏距离（L2范数）
        euclidean_distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1))
        
        # 计算对比损失
        loss_contrastive = torch.mean((1 - label) * 0.5 * euclidean_distance**2 +
                                      label * 0.5 * torch.clamp(self.margin - euclidean_distance, min=0)**2)
        
        return loss_contrastive

class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        
    def forward(self, image_features, text_features):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        labels = torch.eye(num_logits, device=device, dtype=torch.float)
        pred_1 = F.log_softmax(logits_per_image,dim=-1)
        pred_2 = F.log_softmax(logits_per_text,dim=-1)
        loss_a = F.kl_div(pred_1, labels,reduction = 'sum')/num_logits
        loss_b = F.kl_div(pred_2, labels,reduction = 'sum')/num_logits
        total_loss = (loss_a + loss_b)/2
        return total_loss