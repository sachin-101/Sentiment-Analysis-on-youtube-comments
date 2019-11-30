import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class RNNModel(nn.Module):

    def __init__(self, embedding_matrix, hidden_size,  num_layers):
        super(RNNModel, self).__init__()
        self.embedding, num_embeddings, embedding_dim = self.create_embedding_layer(embedding_matrix)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=self.num_layers)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, inp):
        embed_seq = self.embedding(inp).permute(1, 0, 2)
        output, _ = self.lstm(embed_seq)
        y = self.linear(output[-1,...])
        return F.sigmoid(y)

    @staticmethod
    def create_embedding_layer(embed_matrix, trainable=True):
        num_embeddings, embedding_dim = embed_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        emb_layer.load_state_dict({'weight':embed_matrix})
        if trainable:
            emb_layer.weight.requires_grad = True
        return emb_layer, num_embeddings, embedding_dim


if __name__ == "__main__":
    embedding_matrix = torch.rand((50, 10))
    hidden_size = 10
    num_layers = 2
    model = RNNModel(embedding_matrix, hidden_size, num_layers)
    x = torch.arange(20).view((20, 1))
    output = model.forward(x)
    print(output.shape)
    
