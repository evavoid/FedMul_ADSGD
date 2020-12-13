"""
-----------------------------------------------------------------------------------
--  @file       MLfog_lib->modules_base.py
--  @author     Ma Haoming(En:louis)(https://github.com/evavoid)
--  @brief      xxxxxxxxx
    
--  @Ide        PyCharm
--  @time       2020/12/11-23:40
-----------------------------------------------------------------------------------
"""

import torch.nn as nn
import torch
import torch.optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

class LocalMnist(nn.Module):
    def __init__(self, args=None, data=None):
        super(LocalMnist, self).__init__()
        self.args = args
        self.data = DataLoader(data, batch_size=args.bs, shuffle=True, num_workers=0)

        self.network1_linear = nn.Linear(784, 10).to(torch.device('cuda:0'))
        self.network2_relu = nn.ReLU().to(torch.device('cuda:0'))
        self.criterion = nn.MSELoss().to(torch.device('cuda:0'))

        self.loss = None #对于训练集
        self.accuracy = None #对于测试集

        self.weight_grad = None
        self.bias_grad = None
        self.weight_grad_error = 0
        self.bias_grad_error = 0

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.network1_linear(x)
        x = self.network2_relu(x)
        return x

    def start_local_train(self):
        for x_batchs, y_batchs in self.data:
            # 前向传播，
            pred_batchs = self.__call__(x_batchs.to(torch.device('cuda:0')))
            y_batchs = torch.eye(10)[y_batchs, :]
            # 定义损失函数
            self.loss = self.criterion(pred_batchs.to(torch.device('cuda:0')), y_batchs.to(torch.device('cuda:0')))
            # 优化
            optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr)
            optimizer.zero_grad()
            self.loss.backward()
            optimizer.step()
        return self.loss

    def start_compress_grad(self):
        pass
        # self.weight_grad = self.network1_linear.weight.grad
        # self.bias_grad = self.network1_linear.bias.grad
        #
        # self.weight_grad = self.weight_grad + self.weight_grad_error
        # self.bias_grad = self.bias_grad + self.bias_grad_error

    def start_local_test(self, data):
        test_data = DataLoader(data, batch_size=self.args.bs, shuffle=True, num_workers=0)
        self.accuracy = 0
        for x_batchs, y_batchs in test_data:
            pred_batchs = self.__call__(x_batchs.to(torch.device('cuda:0')))
            pred_batchs = torch.topk(pred_batchs, 1)[1].squeeze(1)
            out = [1 if i != 0 else 0 for i in pred_batchs-y_batchs.to(torch.device('cuda:0'))]
            self.accuracy += sum(out)
        self.accuracy /= len(data)
        return self.accuracy
    #     data_points = next(self.data)
    #     pred=self.__call__(self.data_points[0].cuda())
    #     pred = torch.topk(pred, 1)[1].squeeze(1)
    #     image = self.data_points[0].cpu().clone().squeeze(0)
    #     plt.imshow(image)
    #     plt.show()


class GlobalMnist(nn.Module):
    def __init__(self, args=None):
        super(GlobalMnist, self).__init__()
        self.args = args

    def forward(self, x):
        pass

"""
unloader = transforms.ToPILImage()
image = data_points[0].cpu().clone()  # clone the tensor
image = image.squeeze(0)  # remove the fake batch dimension
plt.imshow(image)
plt.show()
print(data_points[1])
"""