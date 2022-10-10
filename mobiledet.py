from timm.models.efficientnet_blocks import *
from tucker_conv.conv import TuckerConv
import torch.nn as nn
import torch


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

        self.n_class = classes
        
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

        #SSDlite
        self.edge12 = Edge_Residual(input=96, output=12, exp_ratio=1, stride=1)
        self.edge13 = Edge_Residual(input=384, output=24, exp_ratio=1, stride=1)
        self.edge14 = Edge_Residual(input=512, output=24, exp_ratio=1, stride=1)
        self.edge15 = Edge_Residual(input=256, output=24, exp_ratio=1, stride=1)
        self.edge16 = Edge_Residual(input=256, output=24, exp_ratio=1, stride=1)
        self.edge17 = Edge_Residual(input=128, output=24, exp_ratio=1, stride=1)


        self.edge18 = Edge_Residual(input=96, output=int(3*(self.n_class+1)), exp_ratio=1, stride=1)
        self.edge19 = Edge_Residual(input=384, output=int(6*(self.n_class+1)), exp_ratio=1, stride=1)
        self.edge20 = Edge_Residual(input=512, output=int(6*(self.n_class+1)), exp_ratio=1, stride=1)
        self.edge21 = Edge_Residual(input=256, output=int(6*(self.n_class+1)), exp_ratio=1, stride=1)
        self.edge22 = Edge_Residual(input=256, output=int(6*(self.n_class+1)), exp_ratio=1, stride=1)
        self.edge23 = Edge_Residual(input=128, output=int(6*(self.n_class+1)), exp_ratio=1, stride=1)

        #classifier
        #self.conv3 = nn.Conv2d(42, 42, 1, 1, 0, bias=False)
        #self.flat = nn.Flatten(1, -1)
        #self.fc1 = nn.Linear(22374, classes)

    
    def forward(self, x):
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
        x32 = self.inv15(x31)#4M 113-1

        if self.net_type == "classifier":

            #x = self.conv2(x)
            x33 = self.flat(x32)
            x34 = self.fc1(x33)
            return x34
        if self.net_type == "detector":
            
            x33 = self.edge12(x23)
            x34 = torch.reshape(x33, (x33.size()[0],int(x33.size()[1]*x33.size()[2]*x33.size()[3]/4),1,4))

            x35 = self.edge13(x28)
            x36 = torch.reshape(x35, (x35.size()[0],int(x35.size()[1]*x35.size()[2]*x35.size()[3]/4),1,4))

            x37 = self.edge14(x29)
            x38 = torch.reshape(x37, (x37.size()[0],int(x37.size()[1]*x37.size()[2]*x37.size()[3]/4),1,4))

            x39 = self.edge15(x30)
            x40 = torch.reshape(x39, (x39.size()[0],int(x39.size()[1]*x39.size()[2]*x39.size()[3]/4),1,4))

            x41 = self.edge16(x31)
            x42 = torch.reshape(x41, (x41.size()[0],int(x41.size()[1]*x41.size()[2]*x41.size()[3]/4),1,4))

            x43 = self.edge17(x32)#4.06M 116-1
            x44 = torch.reshape(x43, (x43.size()[0],int(x43.size()[1]*x43.size()[2]*x43.size()[3]/4),1,4))#4.07M 117-1 all 6 - 8.95M

            x45 = torch.cat((x34,x36,x38,x40,x42,x44), dim=1)#9M 138-1
            #1x1x4x2034 -> 1x2034x4
            x451 = torch.reshape(x45, (x45.size()[0],x45.size()[1],x45.size()[3]))#10M 139-1
            #0,3,2 -> 1,4,1 (1,2034,1,4)


            x46 = self.edge18(x23)
            x47 = torch.reshape(x46, (x46.size()[0], int(x46.size()[1]*x46.size()[2]*x46.size()[3]/(self.n_class+1)), (self.n_class+1)))

            x48 = self.edge19(x28)
            x49 = torch.reshape(x48, (x48.size()[0], int(x48.size()[1]*x48.size()[2]*x48.size()[3]/(self.n_class+1)), (self.n_class+1)))

            x50 = self.edge20(x29)
            x51 = torch.reshape(x50, (x50.size()[0], int(x50.size()[1]*x50.size()[2]*x50.size()[3]/(self.n_class+1)), (self.n_class+1)))

            x52 = self.edge21(x30)
            x53 = torch.reshape(x52, (x52.size()[0], int(x52.size()[1]*x52.size()[2]*x52.size()[3]/(self.n_class+1)), (self.n_class+1)))

            x54 = self.edge22(x31)
            x55 = torch.reshape(x54, (x54.size()[0], int(x54.size()[1]*x54.size()[2]*x54.size()[3]/(self.n_class+1)), (self.n_class+1)))

            x56 = self.edge23(x32)
            x57 = torch.reshape(x56, (x56.size()[0], int(x56.size()[1]*x56.size()[2]*x56.size()[3]/(self.n_class+1)), (self.n_class+1)))

            x58 = torch.cat((x47,x49,x51,x53,x55,x57), dim=1)

            x581 = torch.sigmoid(x58)

            x100 = torch.cat((x451, x581), dim=2)

            #x100 = self.flat(x100)
            #x100 = self.fc1(x100)
            return x100
            #return x23,x28,x29,x30,x31,x32
