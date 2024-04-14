from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class DealDataset(Dataset):

    def __init__(self, file_type: str):
        valid_record = pd.read_csv('data/' + file_type + '.txt', names=list("ABCDEFGHIJK"), sep='\t')
        self.valid_record = valid_record
        # print(valid_record)
        item_sample = np.loadtxt('data/' + file_type + '_item_sampling.txt', delimiter='\t', dtype=int64)
        user_sample = np.loadtxt('data/' + file_type + '_user_sampling.txt', delimiter='\t', dtype=int64)

        self.target_user = torch.from_numpy(valid_record["A"].values)
        self.pos_item = torch.from_numpy(item_sample[:, 0])
        self.pos_user = torch.from_numpy(user_sample[:, 0])
        self.item_sample = torch.from_numpy(item_sample[:, 0:])
        self.user_sample = torch.from_numpy(user_sample[:, 0:])
        self.len = self.target_user.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.target_user[index], self.pos_item[index], self.pos_user[index], self.item_sample[index], \
               self.user_sample[index]

    def load_H(self):
        group_member_dict = {}
        file_path = 'data/Weeplaces/group_user.txt'  # 替换为你的数据集文件路径

        with open(file_path, 'r') as file:
            for line in file:
                group, member = map(int, line.strip().split(','))
                print(group,member)
        #         if group in group_member_dict:
        #             group_member_dict[group].append(member)
        #         else:
        #             group_member_dict[group] = [member]

        # # 获取最大的组序号和成员序号
        # max_group = max(group_member_dict.keys())
        # max_member = max(member for members in group_member_dict.values() for member in members)

        # # 创建二维列表，并初始化为0
        # H = [[0] * (max_group + 1) for _ in range(max_member + 1)]

        # # 填充二维列表
        # for group, members in group_member_dict.items():
        #     for member in members:
        #         H[member][group] = 1

        # # 打印结果
        # for row in H:
        #     print(row)
        return H


H = np.zeros((8643+25081,22733))
file_path = 'data/Weeplaces/group_user.txt'  # 替换为你的数据集文件路径
with open(file_path, 'r') as file:
    for line in file:
        group, member = map(int,map(float, line.strip().split(',')))
        H[member][group] = 1

'''
group_member_dict = {}
index = 0
for row in H.T:
    member = np.nonzero(row)[0].tolist()
    group_member_dict[index] = member
    index+=1
print('-'*89)
print(group_member_dict)
'''
file_path = 'data/Weeplaces/group_item_train.txt'
with open(file_path, 'r') as file:
    for line in file:
        group, item = map(int,line.strip().split(','))
        H[item+8643][group] = 1
print(H[8643+19900][0])