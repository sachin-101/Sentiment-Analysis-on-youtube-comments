import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RNNModel(nn.Module):

    def __init__(self, embedding_matrix, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.embedding, num_embeddings, embedding_dim = self.create_embedding_layer(embedding_matrix)
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=self.num_layers)
    
    def forward(self, inp, hidden):
        return self.lstm(self.embedding(inp), hidden) 

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
    hidden_size = 20
    num_layers = 2
    model = RNNModel(embedding_matrix, hidden_size, num_layers)
    h = torch.rand((num_layers, 1, hidden_size))
    c = torch.rand((num_layers, 1, hidden_size))
    x = torch.arange(5).view((5, 1))
    output, (h, c) = model.forward(x, (h, c))
    