import os
from tqdm.autonotebook import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import mobiledet

# CUDA_VISIBLE_DEVICES="" for CPU run

imagenet_dir = '/media/valdis/NVME/datasets/imagenet/ILSVRC/Data/CLS-LOC'
batch_size = 128
workers = 8
EPOCHS = 100
n_classes = 1000
PATH_TO_SAVE = './runs'

def train(model, dataloaders, loss_fn, optimizer, scheduler, num_epochs = 10):  
    model.to(device)  
    #print(model)

    best_acc = 0.0

    time_beginning = time.time()
    losses = {'train': [], "val": []}
    accuracy = {'train': [], "val": []}
    #log_template = "\nEpoch {ep:03d} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f}"

    #pbar = trange(num_epochs, desc="Epoch")
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            processed_data = 0
            #for inputs, labels in dataloader:
            for data in tqdm(dataloaders[phase], leave=False, desc=f"{phase} iter"):
                inputs, labels = data

                if train_on_gpu:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
 
                if phase == "train":
                    optimizer.zero_grad()

                if phase == "train":
                    outputs = model(inputs)
                else:
                   with torch.no_grad():
                        outputs = model(inputs)                   
                
                loss = loss_fn(outputs, labels)
                preds = torch.argmax(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                processed_data += inputs.size(0)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            accuracy[phase].append(epoch_acc)

            if phase == 'val':
                torch.save(model.state_dict(), PATH_TO_SAVE+"/last.pt")
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), PATH_TO_SAVE+"/best.pt")

        #Print log each epoch
        print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train acc", "{:.4f}".format(accuracy['train'][epoch]), "train loss", "{:.4f}".format(losses['train'][epoch]), 
                                                       "val acc", "{:.4f}".format(accuracy['val'][epoch]), "val loss", "{:.4f}".format((losses['val'][epoch])))
    
    time_elapsed = time.time() - time_beginning
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))



    




#Check if GPU is enable
if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...')
    train_on_gpu = True
    device = torch.device("cuda")
else:
    if torch.backends.mps.is_available():
        print('GPU on M1 MAC is available!  Training on GPU ...')
        train_on_gpu = True
        device = torch.device("mps")
    else:
        print('CUDA and MPS are not available.  Training on CPU ...')


#Loading dataset - like Imagenet, where 1k folders with each classes
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
    ])
imagenet_data = {x: torchvision.datasets.ImageFolder(os.path.join(imagenet_dir, x), transform=data_transforms)
                for x in ['train', 'val']}
data_loaders = {x: torch.utils.data.DataLoader(imagenet_data[x], batch_size=batch_size, shuffle=True, num_workers=workers)
                for x in ['train', 'val']}
dataset_sizes = {x: len(imagenet_data[x]) for x in ['train', 'val']}
class_names = imagenet_data['train'].classes
print(dataset_sizes)
train_features, train_labels = next(iter(data_loaders['train']))


#Configurate model and hyperparameters
num_features = 1280
#model = mobiledet.MobileDetTPU()
model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
model.classifier = nn.Linear(num_features, n_classes)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train(model, data_loaders, loss_fn, optimizer, exp_lr_scheduler, EPOCHS)