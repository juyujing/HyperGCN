import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
from tqdm import tqdm
from model.MGBR import MGBR
from copy import deepcopy
from model.metric import compute_rr, compute_ndcg, compute_recall
from dataset import DealDataset



# user_num = 125012
# item_num = 30516


device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
describe = 'hyperAdjmtl1'
status = 0 # 1==train, 0==test

################# Train and Test ###############################

def test_model(dataset, model, device, epoch):
    item_total_num = 0
    user_total_num = 0
    ks = [1, 5, 10, 20, 50]
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
        # item_sample_score = item_sample_score[:,0:10]
        # user_sample_score = user_sample_score[:,0:10]
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

    f = open("log/log_{}_{}{}.txt".format(args.model, args.remark, describe), "a")
    s0 = "epoch %d ," % epoch
    i05r = "rr@%d: %f" % (ks[1], item_total_rr[1] / item_total_num)
    i10r = "rr@%d: %f" % (ks[2], item_total_rr[2] / item_total_num)
    i20r = "rr@%d: %f" % (ks[3], item_total_rr[3] / item_total_num)
    i50r = "rr@%d: %f" % (ks[4], item_total_rr[4] / item_total_num)
    i05n = "ndcg@%d: %f" % (ks[1], item_total_ndcg[1] / item_total_num)
    i10n = "ndcg@%d: %f" % (ks[2], item_total_ndcg[2] / item_total_num)
    i20n = "ndcg@%d: %f" % (ks[3], item_total_ndcg[3] / item_total_num)
    i50n = "ndcg@%d: %f" % (ks[4], item_total_ndcg[4] / item_total_num)
    print("Test " + s0 + "\t" + i05r + "\t" + i10r + "\t" + i20r + "\t" + i50r)
    print("Test " + s0 + "\t" + i05n + "\t" + i10n + "\t" + i20n + "\t" + i50n)
    f.write(s0 + 'Item Task :' + "\t" + i05r + "\t" + i10r + "\t" + i20r + "\t" + i50r + "\n")
    f.write(s0 + 'Item Task :' + "\t" + i05n + "\t" + i10n + "\t" + i20n + "\t" + i50n + "\n")
    
    u05r = "rr@%d: %f" % (ks[1], user_total_rr[1] / user_total_num)
    u10r = "rr@%d: %f" % (ks[2], user_total_rr[2] / user_total_num)
    u20r = "rr@%d: %f" % (ks[3], user_total_rr[3] / user_total_num)
    u50r = "rr@%d: %f" % (ks[4], user_total_rr[4] / user_total_num)
    u05n = "ndcg@%d: %f" % (ks[1], user_total_ndcg[1] / user_total_num)
    u10n = "ndcg@%d: %f" % (ks[2], user_total_ndcg[2] / user_total_num)
    u20n = "ndcg@%d: %f" % (ks[3], user_total_ndcg[3] / user_total_num)
    u50n = "ndcg@%d: %f" % (ks[4], user_total_ndcg[4] / user_total_num)
    print("Test " + s0 + "\t" + u05r + "\t" + u10r + "\t" + u20r + "\t" + u50r)
    print("Test " + s0 + "\t" + u05n + "\t" + u10n + "\t" + u20n + "\t" + u50n)
    f.write(s0 + 'User Task :' + "\t" + u05r + "\t" + u10r + "\t" + u20r + "\t" + u50r + "\n")
    f.write(s0 + 'User Task :' + "\t" + u05n + "\t" + u10n + "\t" + u20n + "\t" + u50n + "\n")
    f.close()

    return item_total_ndcg[2] / item_total_num, user_total_ndcg[2] / user_total_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GCN-PLE')
    parser.add_argument('-m', '--model', type=str, default='adjmtl', help='model')
    parser.add_argument('-r', '--remark', type=str, default='', help='just a remark')
    parser.add_argument('-s', '--save', type=int, default=0, help='save emb or not')
    args = parser.parse_args()
    
    args.user_num = 125012
    args.item_num = 30516

    print(device0)
    print(device1)
    
    in_dimension, hidden_dimension, out_dimension = 256, 128, 16 # low_para: 128, 64, 16
    model = MGBR(in_dimension, hidden_dimension, out_dimension, device0, args)
    # model.load_state_dict(torch.load("dict/dict_{}_{}4.pt".format(args.model, args.remark)))
    model.to(device0)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6)

    trainDataset = DealDataset("train")
    tuneDataset = DealDataset("tune")
    testDataset = DealDataset("test")
    train_loader = DataLoader(dataset=trainDataset, batch_size=96, shuffle=True)
    tune_loader = DataLoader(dataset=tuneDataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(dataset=testDataset, batch_size=24, shuffle=True)

    incr = 0
    max_item_rr = 0
    max_user_rr = 0
    epoch = 0

    if status == 1:
        for epoch in range(25):
            print('epoch %d Start!'% epoch)
            for data in tqdm(train_loader):
                tu, _, _, item_s, user_s = data
                tu = tu.to(device0)
                item_s, user_s = item_s.to(device0), user_s.to(device0)
                loss, item_sample_score, user_sample_score = model(tu, item_s, user_s)
                loss = loss.unsqueeze(1)
                optimizer.zero_grad()
                loss.backward(torch.ones_like(loss))
                optimizer.step()

            end = time.time()
            s0 = "epoch %d " % epoch
            print(s0 + 'have trained')

            if args.save:
                e_pi, e_up, e_ui = deepcopy(model.embed_pi.weight), deepcopy(model.embed_u.weight), deepcopy(model.embed.weight)
                np.save('embs/{}_{}_pi_emb_{}.npy'.format(args.model, args.remark, epoch), e_pi.detach().cpu().numpy())
                np.save('embs/{}_{}_up_emb_{}.npy'.format(args.model, args.remark, epoch), e_up.detach().cpu().numpy())
                np.save('embs/{}_{}_ui_emb_{}.npy'.format(args.model, args.remark, epoch), e_ui.detach().cpu().numpy())

            f = open("log/log_{}_{}{}.txt".format(args.model, args.remark,describe), "a")
            f.write("tune\n")
            f.close()
            item_rr, user_rr = test_model(tune_loader, model, device0, epoch)
            if item_rr > max_item_rr and user_rr > max_user_rr:
                max_item_rr, max_user_rr = item_rr, user_rr
                torch.save(model.state_dict(), "dict/dict_{}_{}{}.pt".format(args.model, args.remark, describe))
                print('A better model has been saved')
                incr = 0
            else:
                model.load_state_dict(torch.load("dict/dict_{}_{}{}.pt".format(args.model, args.remark, describe)))
                model.to(device0)
        
            t_model = MGBR(in_dimension, hidden_dimension, out_dimension, device0, args)
            t_model.load_state_dict(torch.load("dict/dict_{}_{}{}.pt".format(args.model, args.remark, describe)))
            t_model.to(device0)
        
            f = open("log/log_{}_{}{}.txt".format(args.model, args.remark, describe), "a")
            f.write("test\n")
            f.close()
            item_rr, user_rr = test_model(test_loader, t_model, device0, epoch)
    
    else:
        t_model = MGBR(in_dimension, hidden_dimension, out_dimension, device0, args)
        print("dict/draft/dict_{}_{}{}.pt".format(args.model, args.remark, describe))
        t_model.load_state_dict(torch.load("dict/draft/dict_{}_{}{}.pt".format(args.model, args.remark, describe)))
        t_model.to(device0)
        item_rr, user_rr = test_model(test_loader, t_model, device0, epoch)
    