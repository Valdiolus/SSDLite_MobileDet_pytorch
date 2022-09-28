from timm.models.efficientnet_blocks import *
from tucker_conv.conv import TuckerConv
import torch.nn as nn
import torch


def bottleneck_layer(input, output, exp_ratio, stride):

    middle_dim = int(input * exp_ratio)

    if exp_ratio == 1:
        middle_dim = 32
        conv1 = nn.Conv2d(input, middle_dim, 3, 2, 1, bias=False)
        depth = nn.Conv2d(middle_dim, 8, 1, stride, 0) # no paddings!!!
        conv2 = nn.Conv2d(8, output, 3, 1, 1, bias=False)

        return nn.Sequential(
            conv1,
            nn.BatchNorm2d(middle_dim),
            nn.ReLU6(inplace=True),
            depth,
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            conv2,
            nn.BatchNorm2d(output),
            nn.ReLU6(inplace=True))
    else:
        conv1 = nn.Conv2d(input, middle_dim, 1, 1, 0, bias=False)
        depth = nn.Conv2d(middle_dim, middle_dim, 3, stride, 1, groups=middle_dim)
        conv2 = nn.Conv2d(middle_dim, output, 1, 1, 0, bias=False)
    
        return nn.Sequential(
            conv1,
            nn.BatchNorm2d(middle_dim),
            #nn.ReLU6(inplace=True),
            depth,
            nn.BatchNorm2d(middle_dim),
            nn.ReLU6(inplace=True),
            conv2,
            nn.BatchNorm2d(output),
            #nn.ReLU6(inplace=True)
            )

def Edge_Residual(input, output, exp_ratio, stride, exp_kernel=3):
    middle_dim = int(input * exp_ratio)

    kernel1 = exp_kernel
    stride1 = stride
    if exp_kernel == 5:
        padding1 = 2
    else:
        padding1 = 1
    conv1 = nn.Conv2d(input, middle_dim, kernel1, stride1, padding1, bias=False)

    kernel2 = 1
    stride2 = 1
    padding2 = 0
    conv2 = nn.Conv2d(middle_dim, output, kernel2, stride2, padding2, bias=False)

    return nn.Sequential(
        conv1,
        nn.BatchNorm2d(middle_dim),
        nn.ReLU6(inplace=True),
        conv2,
        nn.BatchNorm2d(output),
        )

def Inverted_Residual(input, output, exp_ratio, stride, exp_kernel=3, last_relu=0):
    middle_dim = int(input * exp_ratio)

    kernel1 = 1
    stride1 = 1
    padding1 = 0
    conv1 = nn.Conv2d(input, middle_dim, kernel1, stride1, padding1, bias=False)

    kernel2 = exp_kernel
    stride2 = stride
    if exp_kernel == 5:
        padding2 = 2
    else:
        padding2 = 1

    depth = nn.Conv2d(middle_dim, middle_dim, kernel2, stride2, padding2, groups=middle_dim)

    kernel3 = 1
    stride3 = 1
    padding3 = 0   
    conv2 = nn.Conv2d(middle_dim, output, kernel3, stride3, padding3, bias=False)

    if last_relu: #make in 1 string?
        return nn.Sequential(
            conv1,
            nn.BatchNorm2d(middle_dim),
            nn.ReLU6(inplace=True),
            depth,
            nn.BatchNorm2d(middle_dim),
            nn.ReLU6(inplace=True),
            conv2,
            nn.BatchNorm2d(output),
            nn.ReLU6(inplace=True)
            )   
    else:
        return nn.Sequential(
            conv1,
            nn.BatchNorm2d(middle_dim),
            nn.ReLU6(inplace=True),
            depth,
            nn.BatchNorm2d(middle_dim),
            nn.ReLU6(inplace=True),
            conv2,
            nn.BatchNorm2d(output)
            )


class MobileDetTPU(nn.Module):
    def __init__(self, net_type, classes):
        self.net_type = net_type
        self.classes = classes
        super(MobileDetTPU, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.rl1 = nn.ReLU6(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(8)
        self.rl2 = nn.ReLU6(inplace=True)
        
        self.edge1 = Edge_Residual(input=8, output=16, exp_ratio=2, stride=1)
        self.edge2 = Edge_Residual(input=16, output=16, exp_ratio=8, stride=2)
        self.edge3 = Edge_Residual(input=16, output=16, exp_ratio=4, stride=1)
        self.edge4 = Edge_Residual(input=16, output=16, exp_ratio=8, stride=1)
        self.edge5 = Edge_Residual(input=16, output=16, exp_ratio=4, stride=1)

        self.edge6 = Edge_Residual(input=16, output=40, exp_ratio=8, stride=2, exp_kernel=5)
        self.edge7 = Edge_Residual(input=40, output=40, exp_ratio=4, stride=1)
        self.edge8 = Edge_Residual(input=40, output=40, exp_ratio=4, stride=1)
        self.edge9 = Edge_Residual(input=40, output=40, exp_ratio=4, stride=1)

        self.inv1 = Inverted_Residual(input=40, output=72, exp_ratio=8, stride=2)
        self.inv2 = Inverted_Residual(input=72, output=72, exp_ratio=8, stride=1)
        self.edge10 = Edge_Residual(input=72, output=72, exp_ratio=4, stride=1)
        self.edge11 = Edge_Residual(input=72, output=72, exp_ratio=4, stride=1)

        self.inv3 = Inverted_Residual(input=72, output=96, exp_ratio=8, stride=1, exp_kernel=5)
        self.inv4 = Inverted_Residual(input=96, output=96, exp_ratio=8, stride=1, exp_kernel=5)
        self.inv5 = Inverted_Residual(input=96, output=96, exp_ratio=8, stride=1)
        self.inv6 = Inverted_Residual(input=96, output=96, exp_ratio=8, stride=1)

        self.inv7 = Inverted_Residual(input=96, output=120, exp_ratio=8, stride=2, exp_kernel=5)
        self.inv8 = Inverted_Residual(input=120, output=120, exp_ratio=8, stride=1)
        self.inv9 = Inverted_Residual(input=120, output=120, exp_ratio=4, stride=1, exp_kernel=5)
        self.inv10 = Inverted_Residual(input=120, output=120, exp_ratio=8, stride=1)

        self.inv11 = Inverted_Residual(input=120, output=384, exp_ratio=8, stride=1, exp_kernel=5)
        self.inv12 = Inverted_Residual(input=384, output=512, exp_ratio=256/384, stride=2, last_relu=1)
        self.inv13 = Inverted_Residual(input=512, output=256, exp_ratio=0.25, stride=2, last_relu=1)
        self.inv14 = Inverted_Residual(input=256, output=256, exp_ratio=0.5, stride=2, last_relu=1)
        self.inv15 = Inverted_Residual(input=256, output=128, exp_ratio=0.25, stride=2, last_relu=1)



        #classifier
        #self.conv2 = nn.Conv2d(128, 1024, 1, 1, 0, bias=False)
        self.flat = nn.Flatten(1, -1)
        self.fc1 = nn.Linear(128, classes)

    
    def forward(self, x):

        if self.net_type == "classifier":

            x1 = self.conv1(x)
            x2 = self.bn1(x1)
            x3 = self.rl1(x2)

            x4 = self.conv2(x3)
            x5 = self.bn2(x4)
            x6 = self.rl2(x5)

            x7 = self.edge1(x6)
            x8 = self.edge2(x7)
            x9 = torch.add(x8, self.edge3(x8))
            x10 = torch.add(x9, self.edge4(x9))
            x11 = torch.add(x10, self.edge5(x10))
            x12 = self.edge6(x11)
            x13 = torch.add(x12, self.edge7(x12))
            x14 = torch.add(x13, self.edge8(x13))
            x15 = torch.add(x14, self.edge9(x14))

            x16 = self.inv1(x15)
            x17 = torch.add(x16, self.inv2(x16))

            x18 = torch.add(x17, self.edge10(x17))
            x19 = torch.add(x18, self.edge11(x18))

            x20 = self.inv3(x19)
            x21 = torch.add(x20, self.inv4(x20))
            x22 = torch.add(x21, self.inv5(x21))
            x23 = torch.add(x22, self.inv6(x22))
            x24 = self.inv7(x23)
            x25 = torch.add(x24, self.inv8(x24))
            x26 = torch.add(x25, self.inv9(x25))
            x27 = torch.add(x26, self.inv10(x26))
            x28 = self.inv11(x27)

            x29 = self.inv12(x28)
            x30 = self.inv13(x29)
            x31 = self.inv14(x30)
            x32 = self.inv15(x31)


            #x = self.conv2(x)
            x33 = self.flat(x32)
            x34 = self.fc1(x33)
            return x34
        else:
            return x23,x28,x29,x30,x31,x32



"""class bottleneck_layer(nn.Module):
    def __init__(self, input, output, expand_ratio, stride):
        super(bottleneck_layer, self).__init__()
        
        self.middle_dim = int(input * expand_ratio)
        self.expand_ratio = expand_ratio
        self.output = output
        
        self.conv11 = nn.Conv2d(input, self.middle_dim, 1, stride, 0, bias=False)
        self.conv1 = nn.Conv2d(self.middle_dim, self.middle_dim, 1, 1, 0, bias=False)
        
        self.depth = nn.Conv2d(self.middle_dim, self.middle_dim, 3, stride, 1)#groups=self.middle_dim

        self.conv2 = nn.Conv2d(self.middle_dim, output, 1, 1, 0, bias=False)

    def forward(self,x):
        if self.expand_ratio == 1:
            x = self.conv11(x)
        else:
            x = self.conv1(x)

        x = nn.BatchNorm2d(self.middle_dim)(x)
        x = nn.ReLU6(inplace=True)(X)
        x = self.depth(x)
        x = nn.BatchNorm2d(self.middle_dim)(x)
        x = nn.ReLU6(inplace=True)(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(self.output)(x)
        x = nn.ReLU6(inplace=True)(x)

        return x"""