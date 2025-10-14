import torch


embedding_vec=torch.load('01aa010d_-0.044503_embedding.pt')
print(embedding_vec)
print(embedding_vec.shape)
print(embedding_vec.dtype)
print(embedding_vec.size())