import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from classroomnet.backbones.models.resnet import resnet18
import time

class SimulatedModule(nn.Module):
    def __init__(self, feature_dim, statistics, distill_dim):
        super().__init__()
        self.mean3d, self.std3d, self.mean2d, self.std2d = statistics

        self.l1 = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 48, kernel_size=1)
        )

        self.l2 = nn.Sequential(
                nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, distill_dim, kernel_size=1),
                nn.ReLU(inplace=True)
        )

        self.l3 = nn.Sequential(
                nn.Conv2d(distill_dim, 48, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 48, kernel_size=1),
                nn.ReLU(inplace=True)
        )

        self.bn1 = nn.BatchNorm2d(48)

        self.bn2 = nn.BatchNorm2d(distill_dim)
    
    def forward(self, x):
        # DN1
        x = self.l1(x)
        x = self.bn1(x) * self.std3d + self.mean3d
        x_to_train = self.l2(x)

        # DN2
        x = self.bn2(x_to_train)
        x = x * self.std2d + self.mean2d
        x = self.l3(x)

        return x, x_to_train


class StudentModule(nn.Module):
    def __init__(self, num_teachers, feature_dim, statistics_list, distill_dim_list): # feature_dim should be 128 with resnet18
        super().__init__()

        self.num_teachers = num_teachers
        self.resnet = resnet18()
        self.sim_module_list = nn.ModuleList([SimulatedModule(feature_dim, statistics_list[i], distill_dim_list[i]) for i in range(self.num_teachers)])

        self.cls1 = nn.Sequential(
            nn.Conv2d(feature_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 48, kernel_size=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # print("STARTED BACKBONE")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        #st = time.time()

        #print(x.shape, 'input shape')
        x = self.resnet(x)
        #print(time.time() - st, 'resnet time')
        #st = time.time()

        #print(x.shape)
        # print("FINISHED BACKBONE RESNET")
        z_list, z_to_train_list = zip(*[sim_module(x) for sim_module in self.sim_module_list])
        x = self.cls1(x)
        z_dict = {'0': x}
        z_dict.update({str(i+1): z_list[i] for i in range(self.num_teachers)})
      
        #print(time.time() - st, 'student for loops time')
        return z_dict, z_to_train_list
