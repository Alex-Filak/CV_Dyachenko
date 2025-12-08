import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3.bias.data.zero_()

        if k == 3:
            self.fc3.weight.data.copy_(torch.eye(3).view(9))
        else:
            self.fc3.weight.data.copy_(torch.eye(64).view(64*64))

    def forward(self, x):
        batck_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = x.view(-1, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Identity transformation

        identity = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(batck_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        return x
