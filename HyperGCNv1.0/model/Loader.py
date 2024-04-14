from torch.utils.data import Dataset, DataLoader
import numpy as np

# class MyDataset(Dataset):
#     def __init__(self, H, batch_size):
#         self.H = H
#         self.batch_size = batch_size
#         self.num_columns = H.shape[1]

#     def __len__(self):
#         return self.num_columns//self.batch_size + 1

#     def __getitem__(self, index):
#         # 随机抽取 batch_size 列，确保不重复
#         selected_columns = np.random.choice(range(self.num_columns), size=self.batch_size, replace=False)
#         # 返回对应列的数据
#         return self.H[:, selected_columns],selected_columns

class MyDataset(Dataset):
    def __init__(self, H, batch_size):
        self.H = H
        self.batch_size = batch_size
        self.num_columns = H.shape[1]
        self.available_columns = list(range(self.num_columns))

    def __len__(self):
        # 计算有多少个完整的批次
        full_batches = self.num_columns // self.batch_size

        # 如果不能整除，增加一个批次
        if self.num_columns % self.batch_size != 0:
            full_batches += 1

        return full_batches

    def __getitem__(self, index):
        # 检查是否有足够的列可供选择
        if len(self.available_columns) < self.batch_size:
            selected_columns = self.available_columns
        else:
            # 随机抽取 batch_size 列，确保不重复
            selected_columns = np.random.choice(self.available_columns, size=self.batch_size, replace=False)

        # 从可用列中移除已经选择的列
        self.available_columns = list(set(self.available_columns) - set(selected_columns))

        # 如果 self.available_columns 为空，重新初始化为所有列
        if not self.available_columns:
            self.available_columns = list(range(self.num_columns))

        # 返回对应列的数据
        return self.H[:, selected_columns]



if __name__ == '__main__':
    # 超图矩阵 H
    # 这里假设 H 是一个 NumPy 数组，你可以根据实际情况替换成你的数据类型
    H = np.random.randint(5, size=(3, 4))  # 假设有100行50列的 H
    print(H)

    # 创建数据集实例
    dataset = MyDataset(H,batch_size=3)
    print(dataset.__len__())

    # 创建 DataLoader 实例
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True)

    # 遍历多个 epochs
    num_epochs = 3
    for epoch in range(num_epochs):
        # 遍历 DataLoader
        for batch in dataloader:
            # 在这里进行你的训练操作，batch 就是每个批次的 H 数据
            print("Epoch {}, Batch shape: {}".format(epoch, batch.shape))
            print(batch)
