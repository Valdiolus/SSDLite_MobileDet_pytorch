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

from torchsummary import summary

# CUDA_VISIBLE_DEVICES="" for CPU run

imagenet_dir = '/Users/valdis/datasets/imagenet/ILSVRC/Data/CLS-LOC'
batch_size = 64
workers = 4
EPOCHS = 150
n_classes = 1000
input_size = 320
PATH_TO_SAVE = './runs'

def train(model, dataloaders, loss_fn, optimizer, scheduler, num_epochs = 10):  
    #count the current date and time
    time_struct = time.gmtime()
    time_now = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)+'-'+str(time_struct.tm_mday)+'_'+str(time_struct.tm_hour+3)+'-'+str(time_struct.tm_min)+'-'+str(time_struct.tm_sec)
    print(time_now)
    train_path = os.path.join(PATH_TO_SAVE, time_now)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    
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
            iter_top1 = 0
            iter_top5 = 0
            correct_top5 = 0
            correct_top1 = 0
            #for inputs, labels in dataloader:
            iter = 0
            with tqdm(dataloaders[phase], leave=False, desc=f"{phase} iter") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}, {phase} iter")
                for data in tepoch:
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

                    #calculate 
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        processed_data += inputs.size(0)

                        iter += 1
                        #if iter % 10 == 0:
                        #    torch.save(model.state_dict(), os.path.join(PATH_TO_SAVE, "tmp.pt"))
                        #    exit(0)
                        tepoch.set_postfix(loss=loss.item(), accuracy=(running_corrects/(batch_size*iter)).item())
                    if phase == "val":
                        iter_top1 = 0
                        iter_top5 = 0

                        iter_top1 += torch.sum(preds == labels.data)
                        iter_top5 += iter_top1

                        for _ in range(4):
                            for i in range(inputs.size(0)):
                                outputs[i,preds[i]] = -1000
                            preds = torch.argmax(outputs, 1)
                            iter_top5 += torch.sum(preds == labels.data)
                        
                        processed_data += inputs.size(0)
                        correct_top1 += iter_top1
                        correct_top5 += iter_top5
                        running_loss += loss.item() * inputs.size(0)

                        tepoch.set_postfix(top1=(correct_top1 / processed_data).item(), top5=(correct_top5 / processed_data).item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            accuracy[phase].append(epoch_acc)

            if phase == 'train':
                scheduler.step()
                
            if phase == 'val':
                correct_top1 = correct_top1.item() / processed_data
                correct_top5 = correct_top5.item() / processed_data
                torch.save(model.state_dict(), os.path.join(train_path, "last.pt"))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(train_path, "best.pt"))

        #Print log each epoch
        print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train acc", "{:.4f}".format(accuracy['train'][epoch]), "train loss", "{:.4f}".format(losses['train'][epoch]), 
                                                       "val loss", "{:.4f}".format((losses['val'][epoch])), "val top5:", correct_top5, "val top1:", correct_top1)
    
    time_elapsed = time.time() - time_beginning
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print("Final top5:", correct_top5, "top1:", correct_top1)

if __name__ == '__main__':
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
        transforms.Resize((input_size,input_size)),
        transforms.CenterCrop(298),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    model = mobiledet.MobileDetTPU(net_type="classifier", classes=n_classes)

    #model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
    #model.classifier = nn.Linear(1280, n_classes)

    #The optimal learning rate of the MobileNet-V2 model is 1.66 × 10 −3 - ???
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), 0.05, momentum=0.9, weight_decay=0.00004)
    loss_fn = nn.CrossEntropyLoss()
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(len(imagenet_data['train'])/batch_size))

    train(model, data_loaders, loss_fn, optimizer, exp_lr_scheduler, EPOCHS)