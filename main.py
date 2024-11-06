import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ACNLayer(nn.Module):
    def __init__(self, num_neurons, dropout_rate=0.2):
        super(ACNLayer, self).__init__()
        self.num_neurons = num_neurons
        self.adjacency_matrix = nn.Parameter(torch.empty(num_neurons, num_neurons))
        init.xavier_uniform_(self.adjacency_matrix)
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        self.batch_norm = nn.BatchNorm1d(num_neurons)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, h, h_prev=None):
        h_next = torch.matmul(h, self.adjacency_matrix.T) + self.bias
        h_next = self.batch_norm(h_next)
        h_next = F.relu(h_next)
        h_next = self.dropout(h_next)
        if h_prev is not None:
            h_next = h_next + h_prev  # Connexion résiduelle avec l'entrée précédente
        return h_next

class ACN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate=0.2):
        super(ACN, self).__init__()
        self.num_layers = num_layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        init.xavier_uniform_(self.input_layer.weight)
        self.input_batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.acn_layers = nn.ModuleList([ACNLayer(hidden_size, dropout_rate) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x):
        h = self.input_layer(x)
        h = self.input_batch_norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        h_list = [h]

        # Passage avant
        for i in range(self.num_layers):
            h = self.acn_layers[i](h, h_list[-1])
            h_list.append(h)

        # Passage arrière (réflexion)
        for i in reversed(range(self.num_layers)):
            h = self.acn_layers[i](h, h_list[i])

        output = self.output_layer(h)
        return output
