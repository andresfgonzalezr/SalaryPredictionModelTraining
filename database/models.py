import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class NeuralSalary(nn.Module):
    def __init__(self, n_entries):
        super(NeuralSalary, self).__init__()
        self.Linear1 = nn.Linear(n_entries, 128)
        self.Linear2 = nn.Linear(128, 128)
        self.Linear3 = nn.Linear(128, 128)
        self.Linear4 = nn.Linear(128, 1)
        self.init_weights()  # new code

    def init_weights(self):
        init.xavier_uniform_(self.Linear1.weight)
        init.xavier_uniform_(self.Linear2.weight)
        init.xavier_uniform_(self.Linear3.weight)
        init.xavier_uniform_(self.Linear4.weight)

    def forward(self, inputs):
        prediction1 = F.relu(input=self.Linear1(inputs))
        prediction2 = F.relu(input=self.Linear2(prediction1))
        prediction3 = F.relu(input=self.Linear3(prediction2))
        prediction_f = self.Linear4(prediction3)

        return prediction_f