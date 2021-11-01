import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.deterministic = True  # For reproducibility
torch.backends.cudnn.benchmark = False


class MLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=[], dropout=0):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList([])
        if (hidden_size == 0) or (hidden_size == ''):
            hidden_size = []
        elif isinstance(hidden_size, int):
            hidden_size = [hidden_size,]
        layer_dims = [input_size, ] + hidden_size + [num_classes, ]
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for i in range(len(self.layers)- 1):
            x = self.layers[i].forward(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.layers[-1].forward(x)
        x = self.dropout(x)

        return x


class MultiMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=[], dropout=0, multi_task_classes=0):
        super(MultiMLP, self).__init__()

        self.layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        if (hidden_size == 0) or (hidden_size == ''):
            hidden_size = []
        elif isinstance(hidden_size, int):
            hidden_size = [hidden_size,]
        layer_dims = [input_size, ] + hidden_size + [num_classes, ]
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        for i in reversed(range(1, len(layer_dims) - 1)):
            self.decoder_layers.append(nn.Linear(layer_dims[i], layer_dims[i -1]))
        self.dropout = nn.Dropout(dropout)
        self.multi_task_classes = multi_task_classes
        if self.multi_task_classes:
            self.multi_task_layer = nn.Linear(layer_dims[-2], multi_task_classes)
        else:
            print('Warning: not using multi-task-class layer!')
            raise ValueError

    def forward(self, x):
        for i in range(len(self.layers)- 1):
            x = self.layers[i].forward(x)
            x = F.relu(x)
            x = self.dropout(x)

        x_sup = self.layers[-1].forward(x)
        x_sup = self.dropout(x_sup)

        x_multitask = self.multi_task_layer.forward(x)
        x_multitask = self.dropout(x_multitask)

        x_AE = x
        for i in range(len(self.decoder_layers)):
            x_AE = self.decoder_layers[i].forward(x_AE)
            x_AE = F.relu(x_AE)
            x_AE = self.dropout(x_AE)

        return x_sup, x_multitask, x_AE
