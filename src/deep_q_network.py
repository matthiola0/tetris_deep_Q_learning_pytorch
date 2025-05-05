import torch.nn as nn

class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        # 第一層線性層：輸入是 4 維特徵，輸出 64 維，搭配 ReLU 啟動函數
        self.conv1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))

        # 第二層線性層：64 維輸入轉成 64 維輸出，再次使用 ReLU
        self.conv2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))

        # 第三層：64 維輸入壓縮成 1 維輸出（即對應一個動作的 Q 值）
        self.conv3 = nn.Sequential(nn.Linear(64, 1))

        # 初始化權重
        self._create_weights()

    def _create_weights(self):
        # 使用 Xavier 均勻初始化來穩定訓練
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # 權重初始化
                nn.init.constant_(m.bias, 0)       # 偏差初始化為 0

    def forward(self, x):
        # 前向傳播流程：依序經過 3 層全連接網路
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
