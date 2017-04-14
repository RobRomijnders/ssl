import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self,cf):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, cf['conv1'], kernel_size=5)
        self.conv2 = nn.Conv2d(cf['conv1'], cf['conv2'], kernel_size=5)
        self.conv2_drop = nn.Dropout2d(cf['keep_prob'])
        L = np.floor((np.floor((28-5)/2+1)-5)/2+1)**2*cf['conv2']
        self.fc1 = nn.Linear(int(L), cf['fc1'])
        self.fc2 = nn.Linear(cf['fc1'], 10)

        self.init_weights()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def init_weights(self):
        # follow http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        for m in self.children():
            if isinstance(m,nn.Linear):
                size = m.weight.size()
                fan_in = size[1]  # number of columns
                scale = np.sqrt(2.0 / (fan_in))
                m.weight.data.uniform_(-scale, scale)
                m.bias.data.uniform_(-scale, scale)
            elif isinstance(m,nn.Conv2d):
                size = m.weight.size()
                fan_in = size[2]*size[3]
                scale = np.sqrt(2.0 / (fan_in))
                m.weight.data.uniform_(-scale, scale)
                m.bias.data.uniform_(-scale, scale)
        return
