import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dgl
import numpy as np
from GCN import GCNNet
from HGCN import HGCN

class NPairLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NPairLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negatives):
        # Calculate cosine similarity between anchor and positive
        similarity_pos = nn.functional.cosine_similarity(anchor, positive, dim=1) / self.temperature
        
        # Calculate cosine similarity between anchor and negatives
        print(anchor.unsqueeze(1))
        print(negatives)
        similarity_neg = nn.functional.cosine_similarity(anchor.unsqueeze(1), negatives, dim=2) / self.temperature
        
        # Calculate N-pair loss
        loss = torch.log(torch.sum(torch.exp(similarity_neg), dim=1)) - similarity_pos
        loss = torch.mean(loss)
        return loss

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ContrastiveModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.fc(x)

# # Generate some example data
# batch_size = 2
# input_dim = 8
# embedding_dim = 4
# num_negatives = 5

# anchor = torch.randn(batch_size, input_dim)
# positive = torch.randn(batch_size, input_dim)
# negatives = torch.randn(batch_size, num_negatives, input_dim)  # 5 negative samples per anchor

# # Create the model and loss
# model = ContrastiveModel(input_dim, embedding_dim)
# criterion = NPairLoss()

# # Forward pass
# anchor_embedding = model(anchor)
# positive_embedding = model(positive)
# negative_embeddings = torch.tensor([model(neg).tolist() for neg in negatives])

# # Calculate the loss
# loss = criterion(anchor_embedding, positive_embedding, negative_embeddings)

# # Backpropagate and update model
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# print("Loss:", loss.item())


if __name__ == '__main__':
    device_id = "cuda:" + str(1)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    # model = HGCN(num_users=8643, num_items=25081, num_groups=22733, in_dim=128,out_dim=32, attention_hidden_dim=64, activation=nn.relu, device=device)
    # print(model)

    model1 = GCNNet(128,64,32,activation=F.relu)
    print(model1)
    model2 = HGCN(num_users=3, num_items=2, num_groups=2, in_dim=128,out_dim=32, attention_hidden_dim=64, activation=F.relu, device=device)
    print(model2)
    feats1 = nn.Embedding(5,128)
    feats2 = nn.Embedding(5,128)
    edges_src = np.random(100)
    edges_dst = np.random(100)
    g = dgl.graph((edges_src, edges_dst),num_nodes=2)
    H = [[1,0],
         [1,1],
         [0,1],
         [1,1],
         [0,1]]
    out1 = model1(g,feats1)
    out2 = model2(H,feats2)
    
