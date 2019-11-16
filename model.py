import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self, embedding_matrix, hidden_size, num_layers):
        super(Model, self).__init__()
        self.embedding, num_embeddings, embedding_dim = self.create_embedding_layer(embedding_matrix) 
        self.num_layers = num_layers
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=self.num_layers, bidirectional=True)
    
    def forward(self, inp, hidden):
        return self.lstm(self.embedding(inp), hidden) 

    def create_embedding_layer(self, embedding_matrix, trainable=True):
        num_embeddings, embedding_dim = embedding_matrix.shape
        emb_layer = nn.Embedding(num_embeddings, embedding_dim)
        emb_layer.load_state_dict({'weight':embedding_matrix})
        if trainable:
            emb_layer.weight.requires_grad = True
        return emb_layer, num_embeddings, embedding_dim