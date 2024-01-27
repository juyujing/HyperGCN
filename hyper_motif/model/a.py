# import torch

# # # 创建一个稠密矩阵 a 和一个稀疏矩阵 b（作为掩码矩阵）
# dense_matrix_a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# sparse_indices_b = torch.tensor([[0, 1], [1, 0], [2, 1]])
# sparse_values_b = torch.tensor([1, 1, 1])
# sparse_matrix_b = torch.sparse.FloatTensor(sparse_indices_b.t(), sparse_values_b, dense_matrix_a.size())
# # print(type(sparse_matrix_b))
# c = torch.sparse.mm(sparse_matrix_b,dense_matrix_a)
# print(type(c))
# print(c.shape)

# # 获取稀疏矩阵 b 中的非零元素位置
# sparse_indices_new = sparse_indices_b

# # 获取相应位置的稠密矩阵 a 的值
# sparse_values_new = dense_matrix_a[sparse_indices_new[:, 0], sparse_indices_new[:, 1]]

# # 创建新的稀疏矩阵
# sparse_matrix_new = torch.sparse.FloatTensor(sparse_indices_new.t(), sparse_values_new, dense_matrix_a.size())

# print("稠密矩阵 a:")
# print(dense_matrix_a)

# print(sparse_matrix_b._values()[0])
# sparse_matrix_b._values()[0] = 3

# print("\n稀疏矩阵 b（掩码矩阵）:")
# print(sparse_matrix_b.to_dense())

# print("\n重构的稀疏矩阵:")
# print(sparse_matrix_new.to_dense())


# print(torch.cuda.is_available())


# import torch
# import torch.nn as nn
# from torch.nn.parallel import DataParallel

# # 检查是否有可用的 GPU
# if torch.cuda.is_available():
#     # 获取可见的 GPU 设备数
#     device_count = torch.cuda.device_count()
    
#     # 创建模型
#     model = YourModel()

#     # 将模型包装在 DataParallel 中
#     model = DataParallel(model)

#     # 将模型移动到 GPU 上
#     model = model.cuda()

#     # 创建一个 Tensor 在 CPU 上
#     cpu_tensor = torch.randn(3, 3)

#     # 将 Tensor 从 CPU 移动到 CUDA 设备
#     cuda_tensor = cpu_tensor.to(device='cuda')

#     # 在你的训练循环中使用 model 进行前向和后向传播
#     # 例如：
#     # for inputs, labels in dataloader:
#     #     inputs, labels = inputs.cuda(), labels.cuda()
#     #     cuda_tensor = cpu_tensor.to(device='cuda')
#     #     outputs = model(cuda_tensor)
#     #     loss = criterion(outputs, labels)
#     #     loss.backward()
#     #     optimizer.step()
# else:
#     print("CUDA is not available.")



# import torch
# import torch.nn as nn

# class SimpleModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(SimpleModel, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

# if __name__ == '__main__':
#     input_size = 128
#     hidden_size = 64
#     output_size = 32

#     model = SimpleModel(input_size, hidden_size, output_size)
    
#     # 指定使用的 CUDA 设备
#     device_ids = [0, 2, 3]
#     # 构建设备字符串
#     devices = [f"cuda:{device_id}" for device_id in device_ids]

#     # 将模型和数据移动到指定的 CUDA 设备上
#     model = nn.DataParallel(model, device_ids=device_ids).cuda()

#     # 生成测试数据
#     x = torch.rand((1000, input_size)).cuda()

#     # 定义损失函数和优化器
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#     # 模型训练
#     for epoch in range(5):  # 增加训练轮数
#         optimizer.zero_grad()
#         output = model(x)
#         target = torch.rand_like(output).cuda()
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

#     print("Training completed.")


# import torch

# # 创建一个张量
# tensor_example = torch.tensor([[0, 1, 0],
#                                [2, 0, 3],
#                                [0, 4, 5]])

# # 使用 torch.nonzero 获取非零元素的索引
# nonzero_indices = torch.nonzero(tensor_example)

# print(nonzero_indices)


import torch

# 假设你有一个包含负无穷值的张量 tensor
tensor = torch.tensor([[-float('inf'), -float('inf'), -float('inf')],
                       [4.0, 5.0, -float('inf')]])

# 找到整行都是负无穷的行，并将这些行置为0
row_all_inf_mask = torch.all(tensor == -float('inf'), dim=1)
tensor[row_all_inf_mask, :] = 0.0

# 输出结果
print(tensor)
