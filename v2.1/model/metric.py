import torch
import numpy as np

def compute_rr(inputs: torch.Tensor, ks:list) -> list:
    # print(inputs.shape) torch.Size([24, 1000])
    rr_k = [0 for _ in range(len(ks))]
    for j in range(len(ks)):
        if ks[j] > inputs.shape[1]:
            cutout = inputs
        else:
            cutout = inputs[:, 0:ks[j]]
        for i in range(cutout.shape[0]):
            test_sample = cutout[i]
            a = torch.argsort(test_sample, descending=True) # 返回的是排好序的索引
            b = torch.argsort(a)
            rank = b[0].item()
            if rank < ks[j]:
                rr_k[j] += 1 / (rank + 1)
    return rr_k

def compute_recall(inputs: torch.Tensor, ks: list) -> list:
    recall_k = [0 for _ in range(len(ks))]
    for i in range(inputs.shape[0]):
        test_sample = inputs[i]
        a = torch.argsort(test_sample, descending=True)
        b = torch.argsort(a)
        rank = b[0].item()
        for j in range(len(ks)):
            if rank < ks[j]:
                recall_k[j] += 1
    return recall_k

def compute_hit(inputs: torch.Tensor, ks: list) -> list:
    hit_k = [0.0 for _ in range(len(ks))]
    for j in range(len(ks)):
        if ks[j] > inputs.shape[1]:
            cutout = inputs
        else:
            cutout = inputs[:, 0:ks[j]]
        for i in range(cutout.shape[0]):
            test_sample = cutout[i]
            a = torch.argsort(test_sample, descending=True) # 返回的是排好序的索引
            b = torch.argsort(a) # 返回排名序列
            rank = b[0].item() # 返回索引最小值的排名，也就是第一个正样本的排名
            if rank < ks[j]: # 如果正样本在前k个，则计算它的ndcg值
                hit_k[j] += 1.0
    return hit_k

def compute_ndcg(inputs: torch.Tensor, ks: list) -> list:
    ndcg_k = [0 for _ in range(len(ks))]
    for j in range(len(ks)):
        if ks[j] > inputs.shape[1]:
            cutout = inputs
        else:
            cutout = inputs[:, 0:ks[j]]
        for i in range(cutout.shape[0]):
            test_sample = cutout[i]
            a = torch.argsort(test_sample, descending=True) # 返回的是排好序的索引
            b = torch.argsort(a) # 返回排名序列
            rank = b[0].item() # 返回索引最小值的排名，也就是第一个正样本的排名
            if rank < ks[j]: # 如果正样本在前k个，则计算它的ndcg值
                ndcg_k[j] += 1 / np.log2(rank + 2)
    return ndcg_k


def ndcg_binary_at_k_batch_torch(X_pred, heldout_batch, k=10, device='cpu'):
    """
    Normalized Discounted Cumulative Gain@k for for predictions [B, I] and ground-truth [B, I], with binary relevance.
    ASSUMPTIONS: all the 0's in heldout_batch indicate 0 relevance.
    """
    batch_users = X_pred.shape[0]  # batch_size
    _, idx_topk = torch.topk(X_pred, k, dim=1, sorted=True)
    tp = 1. / torch.log2(torch.arange(2, k + 2, device=device).float())
    heldout_batch_nonzero = (heldout_batch > 0).float()
    DCG = (heldout_batch_nonzero[torch.arange(batch_users, device=device).unsqueeze(1), idx_topk] * tp).sum(dim=1)
    heldout_nonzero = (heldout_batch > 0).sum(dim=1)  # num. of non-zero items per batch. [B]
    IDCG = torch.tensor([(tp[:min(n, k)]).sum() for n in heldout_nonzero]).to(device)
    return torch.sum(DCG /IDCG)/batch_users

def recall_at_k_batch_torch(X_pred, heldout_batch, k=100):
    """
    Recall@k for predictions [B, I] and ground-truth [B, I].
    """
    batch_users = X_pred.shape[0]
    _, topk_indices = torch.topk(X_pred, k, dim=1, sorted=False)  # [B, K]
    X_pred_binary = torch.zeros_like(X_pred)
    if torch.cuda.is_available():
        X_pred_binary = X_pred_binary.cuda()
    X_pred_binary[torch.arange(batch_users).unsqueeze(1), topk_indices] = 1
    X_true_binary = (heldout_batch > 0).float()  # .toarray() #  [B, I]
    k_tensor = torch.tensor([k], dtype=torch.float32)
    if torch.cuda.is_available():
        X_true_binary = X_true_binary.cuda()
        k_tensor = k_tensor.cuda()
    tmp = (X_true_binary * X_pred_binary).sum(dim=1).float()
    recall = tmp / torch.min(k_tensor, X_true_binary.sum(dim=1).float())
    return recall


'''
class TopkMetric(Metric):
    """Base class for top-K based metric"""

    def __init__(self, topk: int):
        super().__init__()
        self.topk = topk
        self.eps = 1e-8

    def __str__(self) -> str:
        return f'{self.__class__.__name__}@{self.topk}'

class Recall(TopkMetric):
    """Recall = TP / (TP + FN)"""

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)
        # ignore row without positive result
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += cast(Number, (is_hit / (num_pos + self.eps)).sum().item())


class NDCG(TopkMetric):
    """Normalized Discounted Cumulative Gain"""

    def DCG(self, hit: torch.Tensor, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        hit = hit / torch.log2(torch.arange(
            2, self.topk+2, device=device, dtype=torch.float))
        return hit.sum(-1)

    def _IDCG(self, num_pos: int) -> Number:
        if num_pos == 0:
            return 1
        else:
            hit = torch.zeros(self.topk, dtype=torch.float)
            hit[:num_pos] = 1
            return self.DCG(hit).item()

    def __init__(self, topk: int) -> None:
        super().__init__(topk)
        self.IDCGs: torch.Tensor = torch.FloatTensor(
            [self._IDCG(i) for i in range(0, self.topk + 1)])

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        num_pos = ground_truth.sum(dim=1).clamp(0, self.topk).long()
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs[num_pos]
        ndcg = dcg / idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += cast(Number, ndcg.sum().item())


class MRR(TopkMetric):
    """Mean Reciprocal Rank = 1 / position(1st hit)"""

    def __init__(self, topk: int):
        super().__init__(topk)
        self.denominator = torch.arange(1, self.topk + 1, dtype=torch.float)
        self.denominator.unsqueeze_(0)

    def __call__(self, scores: torch.Tensor, ground_truth: torch.Tensor) -> None:
        device = scores.device
        is_hit = _get_is_hit(scores, ground_truth, self.topk)
        is_hit /= self.denominator.to(device)
        first_hit_rr = is_hit.max(dim=1)[0]
        num_pos = ground_truth.sum(dim=1)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += cast(Number, first_hit_rr.sum().item())
'''