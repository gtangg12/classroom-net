import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from classroomnet.backbones.models.resnet import resnet18

class SimulatedModule(nn.Module):
    def __init__(self, feature_dim, statistics):
        super().__init__()
        self.mean3d, self.std3d, self.mean2d, self.std2d = statistics

        self.l1 = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, kernel_size=1)
        )

        self.l2 = nn.Sequential(
                nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=1),
                nn.ReLU(inplace=True)
        )

        self.l3 = nn.Sequential(
                nn.Conv2d(96, 96, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.Conv2d(96, 96, kernel_size=1),
                nn.ReLU(inplace=True)
        )

        self.bn1 = nn.BatchNorm2d(96)

        self.bn2 = nn.BatchNorm2d(96)
    
    def forward(self, x):
        # DN1
        x = self.l1(x)
        x = self.bn1(x) * self.std3d + self.mean3d
        x_to_train = self.l2(x)

        # DN2
        x = self.bn2(x_to_train)
        x = self.bn1(x) * self.std2d + self.mean2d
        x = self.l3(x)

        return x, x_to_train


class StudentModule(nn.Module):
    def __init__(self, num_teachers, feature_dim, statistics_list): # feature_dim should be 128 with resnet18
        super().__init__()

        self.num_teachers = num_teachers
        self.resnet = resnet18()
        self.sim_module_list = nn.ModuleList([SimulatedModule(feature_dim, statistics_list[i]) for i in range(self.num_teachers)])

        self.cls1 = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 96, kernel_size=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # print("STARTED BACKBONE")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)

        print(x.shape, 'input shape')
        x = self.resnet(x)
        print(x.shape)
        # print("FINISHED BACKBONE RESNET")
        z_list, z_to_train_list = zip(*[sim_module(x) for sim_module in self.sim_module_list])
        x = self.cls1(x)
        z_dict = {'0': x}
        z_dict.update({str(i+1): z_list[i] for i in range(self.num_teachers)})

        return z_dict, z_to_train_list
