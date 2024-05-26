import os
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DealDataset
from model.HMGBR import HMGBR
from model.MGBR import MGBR
from model.metric import compute_rr, compute_ndcg, compute_recall, compute_hit, ndcg_binary_at_k_batch_torch
from generate import generate

####################### Train and Test #########################

def test_model(dataset, model, device, epoch, mode):
    item_total_num = 0
    user_total_num = 0
    ks = [1, 5, 10, 20, 50]
    item_total_rr = [0 for _ in range(len(ks))]
    item_total_ndcg = [0 for _ in range(len(ks))]
    item_total_recall = [0 for _ in range(len(ks))]
    item_total_hit = [0 for _ in range(len(ks))]
    user_total_rr = [0 for _ in range(len(ks))]
    user_total_ndcg = [0 for _ in range(len(ks))]
    user_total_recall = [0 for _ in range(len(ks))]
    user_total_hit = [0 for _ in range(len(ks))]

    i = 0
    
    for i, data in tqdm(enumerate(dataset)):
        tu, _, _, item_s, user_s = data
        tu = tu.to(device)
        item_s, user_s = item_s.to(device), user_s.to(device)
        _, item_sample_score, user_sample_score = model(tu, item_s, user_s)
        # item_sample_score = item_sample_score[:,0:10]
        # user_sample_score = user_sample_score[:,0:10]
        if mode == 'tune':
            item_total_num = 43893
            user_total_num = 43893
        elif mode == 'test':
            item_total_num = 24130
            user_total_num = 24130
        item_rrs = compute_rr(item_sample_score, ks)
        item_ndcgs = [0,0,0,0,0]
        true_value = torch.zeros(item_sample_score.shape)
        true_value[:, 0] = 1
        true_value = true_value.to('cuda:1')
        # x =0
        # for k in ks:
        #     item_ndcgs[x] = ndcg_binary_at_k_batch_torch(X_pred=item_sample_score, heldout_batch=true_value,k=k, device='cuda:1')
        #     x+=1
        item_ndcgs = compute_ndcg(item_sample_score, ks)
        item_recalls = compute_recall(item_sample_score, ks)
        item_hits = compute_hit(item_sample_score, ks)

        # user_total_num = user_sample_score.shape[0]
        user_rrs = compute_rr(user_sample_score, ks)
        user_ndcgs = [0,0,0,0,0]
        true2_value = torch.zeros(item_sample_score.shape)
        true2_value[:, 0] = 1
        true2_value = true2_value.to('cuda:1')
        # x = 0
        # for k in ks:
        #     user_ndcgs[x] = ndcg_binary_at_k_batch_torch(X_pred=user_sample_score, heldout_batch=true2_value, k=k, device='cuda:1')
        #     x+=1
        user_ndcgs = compute_ndcg(user_sample_score, ks)
        user_recalls = compute_recall(user_sample_score, ks)
        user_hits = compute_hit(user_sample_score, ks)

        for j in range(len(ks)):
            item_total_rr[j] += item_rrs[j]
            item_total_ndcg[j] += item_ndcgs[j]
            item_total_recall[j] += item_recalls[j]
            item_total_hit[j] += item_hits[j]
            user_total_rr[j] += user_rrs[j]
            user_total_ndcg[j] += user_ndcgs[j]
            user_total_recall[j] += user_recalls[j]
            user_total_hit[j] += user_hits[j]

    if args.test and args.load!='':
        f = open("log/{}_{}.txt".format(args.load, args.tag), "a")
    else:
        f = open("log/log_{}_{}.txt".format(args.model, args.tag), "a")
    s0 = "epoch " + str(epoch) + ", "
    i05r = "rr@%d: %f" % (ks[1], item_total_rr[1] / item_total_num)
    i10r = "rr@%d: %f" % (ks[2], item_total_rr[2] / item_total_num)
    i20r = "rr@%d: %f" % (ks[3], item_total_rr[3] / item_total_num)
    i50r = "rr@%d: %f" % (ks[4], item_total_rr[4] / item_total_num)
    i05n = "ndcg@%d: %f" % (ks[1], item_total_ndcg[1] / i)
    i10n = "ndcg@%d: %f" % (ks[2], item_total_ndcg[2] / i)
    i20n = "ndcg@%d: %f" % (ks[3], item_total_ndcg[3] / i)
    i50n = "ndcg@%d: %f" % (ks[4], item_total_ndcg[4] / i)
    i05l = "recall@%d: %f" % (ks[1], item_total_recall[1] / item_total_num)
    i10l = "recall@%d: %f" % (ks[2], item_total_recall[2] / item_total_num)
    i20l = "recall@%d: %f" % (ks[3], item_total_recall[3] / item_total_num)
    i50l = "recall@%d: %f" % (ks[4], item_total_recall[4] / item_total_num)
    i05h = "hit@%d: %f" % (ks[1], item_total_hit[1] / item_total_num)
    i10h = "hit@%d: %f" % (ks[2], item_total_hit[2] / item_total_num)
    i20h = "hit@%d: %f" % (ks[3], item_total_hit[3] / item_total_num)
    i50h = "hit@%d: %f" % (ks[4], item_total_hit[4] / item_total_num)
    print("Test " + s0 + "\t" + i05r + "\t\t" + i10r + "\t\t" + i20r + "\t\t" + i50r)
    print("Test " + s0 + "\t" + i05n + "\t" + i10n + "\t" + i20n + "\t" + i50n)
    # print("Test " + s0 + "\t" + i05l + "\t" + i10l + "\t" + i20l + "\t" + i50l)
    # print("Test " + s0 + "\t" + i05h + "\t\t" + i10h + "\t\t" + i20h + "\t\t" + i50h)
    f.write(s0 + 'Item Task :' + "\t" + i05r + "\t\t" + i10r + "\t\t" + i20r + "\t\t" + i50r + "\n")
    f.write(s0 + 'Item Task :' + "\t" + i05n + "\t" + i10n + "\t" + i20n + "\t" + i50n + "\n")
    # f.write(s0 + 'Item Task :' + "\t" + i05l + "\t" + i10l + "\t" + i20l + "\t" + i50l + "\n")
    # f.write(s0 + 'Item Task :' + "\t" + i05h + "\t\t" + i10h + "\t\t" + i20h + "\t\t" + i50h + "\n")
    
    u05r = "rr@%d: %f" % (ks[1], user_total_rr[1] / user_total_num)
    u10r = "rr@%d: %f" % (ks[2], user_total_rr[2] / user_total_num)
    u20r = "rr@%d: %f" % (ks[3], user_total_rr[3] / user_total_num)
    u50r = "rr@%d: %f" % (ks[4], user_total_rr[4] / user_total_num)
    u05n = "ndcg@%d: %f" % (ks[1], user_total_ndcg[1] / i)
    u10n = "ndcg@%d: %f" % (ks[2], user_total_ndcg[2] / i)
    u20n = "ndcg@%d: %f" % (ks[3], user_total_ndcg[3] / i)
    u50n = "ndcg@%d: %f" % (ks[4], user_total_ndcg[4] / i)
    u05l = "recall@%d: %f" % (ks[1], user_total_recall[1] / user_total_num)
    u10l = "recall@%d: %f" % (ks[2], user_total_recall[2] / user_total_num)
    u20l = "recall@%d: %f" % (ks[3], user_total_recall[3] / user_total_num)
    u50l = "recall@%d: %f" % (ks[4], user_total_recall[4] / user_total_num)
    u05h = "hit@%d: %f" % (ks[1], user_total_hit[1] / user_total_num)
    u10h = "hit@%d: %f" % (ks[2], user_total_hit[2] / user_total_num)
    u20h = "hit@%d: %f" % (ks[3], user_total_hit[3] / user_total_num)
    u50h = "hit@%d: %f" % (ks[4], user_total_hit[4] / user_total_num)
    print("Test " + s0 + "\t" + u05r + "\t\t" + u10r + "\t\t" + u20r + "\t\t" + u50r)
    print("Test " + s0 + "\t" + u05n + "\t" + u10n + "\t" + u20n + "\t" + u50n)
    # print("Test " + s0 + "\t" + u05l + "\t" + u10l + "\t" + u20l + "\t" + u50l)
    # print("Test " + s0 + "\t" + u05h + "\t\t" + u10h + "\t\t" + u20h + "\t\t" + u50h)
    f.write(s0 + 'User Task :' + "\t" + u05r + "\t\t" + u10r + "\t\t" + u20r + "\t\t" + u50r + "\n")
    f.write(s0 + 'User Task :' + "\t" + u05n + "\t" + u10n + "\t" + u20n + "\t" + u50n + "\n")
    # f.write(s0 + 'User Task :' + "\t" + u05l + "\t" + u10l + "\t" + u20l + "\t" + u50l + "\n")
    # f.write(s0 + 'User Task :' + "\t" + u05h + "\t\t" + u10h + "\t\t" + u20h + "\t\t" + u50h + "\n")
    f.close()

    return item_total_ndcg[1] / item_total_num, user_total_ndcg[1] / user_total_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HyperGCN-MTL - A HyperGraph-based MTL model for Group Recommendation.', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-t', '--test', action='store_true', help='set test mode')
    parser.add_argument('-m', '--model', type=str, default='HMGBR', help='''model to use for graph-based learning.
Options:
    - HMGBR(default): HyperGCN + AdjMTL
    - HMGBR-A: HyperGCN + MTL
    - MGBR: Group Buying Recommendation Model Based on Multi-task Learning
''')
    parser.add_argument('-D', '--dataset', type=str, default='beibei', help='''Options:
    - beibei(default)
    - weeplaces
''')
    parser.add_argument('-l', '--load', type=str, default='', help='load model')
    parser.add_argument('-g', '--generate', type=int, default=999, help='number of negative samples in item_sampling and user_sampling')
    parser.add_argument('-x', '--xrate', type=int, default=99, help='number of negative samples at training')
    parser.add_argument('-y', '--yrate', type=int, default=999, help='number of negative samples at testing')
    parser.add_argument('-e', '--epoch', type=int, default=25, help='training epoch(s)')
    parser.add_argument('-d', '--device', type=str, default='0', help='cpu or cuda device serial number')
    parser.add_argument('-T', '--tag', type=str, default='', help='suffix tag')
    args = parser.parse_args()

    args.user_num = 125012
    args.item_num = 30516

    if args.device != 'cpu':
        device0 = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    else:
        device0 = torch.device("cpu")
    print('Running device: ' + str(device0) + '\n')
    
    if not os.path.exists('data/'+args.dataset+'/train_user_sampling.txt'):
        generate(args.dataset, 'train', args.generate, args)
        generate(args.dataset, 'tune', args.generate, args)
        generate(args.dataset, 'test', args.generate, args)

    trainDataset = DealDataset(args.dataset, args.xrate, args.yrate, "train")
    tuneDataset = DealDataset(args.dataset, args.xrate, args.yrate, "tune")
    testDataset = DealDataset(args.dataset, args.xrate, args.yrate, "test")
    train_loader = DataLoader(dataset=trainDataset, batch_size=64, shuffle=True)
    tune_loader = DataLoader(dataset=tuneDataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(dataset=testDataset, batch_size=24, shuffle=True)

    if args.model[:5] == 'HMGBR':
        print(True)
        in_dimension, hidden_dimension, out_dimension = 256, 128, 16
        model = HMGBR(in_dimension, hidden_dimension, out_dimension, device0, args)
        optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6) # weight_decay=1e-7
    elif args.model == 'MGBR':
        in_dimension, hidden_dimension, out_dimension = 128, 64, 16 # low_para: 128, 64, 16 ori_para: 128, 16, 10
        model = MGBR(in_dimension, hidden_dimension, out_dimension, device0, args)
        optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-6) 
    else:
        print('No such model')
        exit(0)
    
    if not args.test:
        if args.load!= '':
            model.load_state_dict(torch.load("dict/{}.pt".format(args.load)))
        model.to(device0)
        
        for epoch in range(args.epoch):
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
            
            if args.epoch<10 or epoch==0 or epoch>=5:
                f = open("log/log_{}_{}.txt".format(args.model, args.tag), "a")
                f.write("tune\n")
                f.close()
                item_rr, user_rr = test_model(tune_loader, model, device0, epoch)
                torch.save(model.state_dict(), "dict/dict_{}_{}.pt".format(args.model, args.tag))
                print('A better model has been saved')
                if args.model == 'HMGBR' or args.model == 'HMGBR-A':
                    t_model = HMGBR(in_dimension, hidden_dimension, out_dimension, device0, args)
                elif args.model == 'MGBR':
                    t_model = MGBR(in_dimension, hidden_dimension, out_dimension, device0, args)
                else:
                    print('No such model')
                    exit(0)
                t_model.load_state_dict(torch.load("dict/dict_{}_{}.pt".format(args.model, args.tag)))
                t_model.to(device0)
                f = open("log/log_{}_{}.txt".format(args.model, args.tag), "a")
                f.write("test\n")
                f.close()
                item_rr, user_rr = test_model(test_loader, t_model, device0, epoch, 'tune')

    else:
        print("Evaluating model: {}".format(args.model))
        loaded_dict = torch.load("dict/{}.pt".format(args.load), map_location=device0)
        model.load_state_dict(loaded_dict)
        model.to(device0)
        # for i, data in tqdm(enumerate(test_loader)):
        #     tu, _, _, item_s, user_s = data
        #     tu = tu.to(device0)
        #     item_s, user_s = item_s.to(device0), user_s.to(device0)
        #     _, item_sample_score, user_sample_score = model(tu, item_s, user_s)
        #     f = open("log/{}_result.txt".format(args.model), "a")
        #     f.write('item_sample_score\n')
        #     f.write(str(item_sample_score))
        #     f.write('item_sample_score\n')
        #     f.write(str(item_sample_score))
        #     f.close()
        item_ndcg, user_ndcg = test_model(test_loader, model, device0, 'test', 'test')
