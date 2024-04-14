import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import model.GCN
import model.HGCN
import dataset
from config import Config
from model.CL import NpairLoss
from hyper_motif.model.metric import compute_rr, compute_ndcg, compute_recall


def test_model(dataset, model, device, epoch):
    item_total_num = 0
    user_total_num = 0
    ks = Config.topK
    item_total_rr = [0 for i in range(len(ks))]
    item_total_recall = [0 for i in range(len(ks))]
    item_total_ndcg = [0 for i in range(len(ks))]
    user_total_rr = [0 for i in range(len(ks))]
    user_total_recall = [0 for i in range(len(ks))]
    user_total_ndcg = [0 for i in range(len(ks))]

    for i, data in tqdm(enumerate(dataset)):

        tu, _, _, item_s, user_s = data
        tu = tu.to(device)
        item_s, user_s = item_s.to(device), user_s.to(device)
        loss, item_sample_score, user_sample_score = model(tu, item_s, user_s)

        item_total_num += item_sample_score.shape[0]
        item_rrs = compute_rr(item_sample_score, ks)
        item_recalls = compute_recall(item_sample_score, ks)
        item_ndcgs = compute_ndcg(item_sample_score, ks)

        user_total_num += user_sample_score.shape[0]
        user_rrs = compute_rr(user_sample_score, ks)
        user_recalls = compute_recall(user_sample_score, ks)
        user_ndcgs = compute_ndcg(user_sample_score, ks)

        for i in range(len(ks)):
            item_total_rr[i] += item_rrs[i]
            item_total_recall[i] += item_recalls[i]
            item_total_ndcg[i] += item_ndcgs[i]
            user_total_rr[i] += user_rrs[i]
            user_total_recall[i] += user_recalls[i]
            user_total_ndcg[i] += user_ndcgs[i]

    f = open("log/log_{}_{}.txt".format(args.model, args.remark), "a")
    s0 = "epoch %d ," % epoch
    s2 = "rr@%d:%f" % (ks[1], item_total_rr[1] / item_total_num)
    s3 = "ndcg@%d:%f" % (ks[1], item_total_ndcg[1] / item_total_num)
    s4 = "rr@%d:%f" % (ks[2], item_total_rr[2] / item_total_num)
    s5 = "ndcg@%d:%f" % (ks[2], item_total_ndcg[2] / item_total_num)
    print("Test " + s0 + "\t" + s2 + "\t" + s3 + "\t" + s4 + "\t" + s5)
    f.write(s0 + 'Item Task :' + "\t" + s2 + "\t" + s3 + "\t" + s4 + "\t" + s5 + "\n")
    s2 = "rr@%d:%f" % (ks[1], user_total_rr[1] / user_total_num)
    s3 = "ndcg@%d:%f" % (ks[1], user_total_ndcg[1] / user_total_num)
    s4 = "rr@%d:%f" % (ks[2], user_total_rr[2] / user_total_num)
    s5 = "ndcg@%d:%f" % (ks[2], user_total_ndcg[2] / user_total_num)
    print("Test " + s0 + "\t" + s2 + "\t" + s3 + "\t" + s4 + "\t" + s5)
    f.write(s0 + 'User Task :' + "\t" + s2 + "\t" + s3 + "\t" + s4 + "\t" + s5 + "\n")
    f.close()

    return item_total_ndcg[2] / item_total_num, user_total_ndcg[2] / user_total_num


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Training in CUDA "+str(Config.gpu_id))
    else:
        print("Training in CPU")
    device_id = "cuda:" + str(Config.gpu_id)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    