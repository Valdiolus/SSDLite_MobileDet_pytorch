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
import wandb

# CUDA_VISIBLE_DEVICES="" for CPU run

#imagenet_dir = '/Users/valdis/datasets/imagenet/ILSVRC/Data/CLS-LOC'
batch_size = 32
workers = 4
EPOCHS = 150
n_classes = 1000
input_size = 320
init_lr = 0.05
init_momentum = 0.9
init_weight_decay = 0.00004
CenterCrop = 320 # 224? was 298 

load_from_file = 0
saved_model = '2022-11-6_21-50-46'
wandb_saved_id = 0

PATH_TO_SAVE = './runs'

wandb_log_interval = 10
wandb_config = {"batch_size": batch_size,
                "num_workers": workers,
                "input size": input_size,
                "epochs": EPOCHS,
                "pin_memory": False,  
                "precision": 32,
                "optimizer": "SGD",
                "lr": init_lr,
                "momentum": init_momentum,
                "weight_decay": init_weight_decay,
                "CenterCrop": CenterCrop,
                }


def train(model, dataloaders, loss_fn, optimizer, scheduler, num_epochs = 10):  
    if load_from_file:
        print("Resume training:", saved_model)
        train_path = os.path.join(PATH_TO_SAVE, saved_model)
    else:
        #count the current date and time
        time_struct = time.gmtime()
        time_now = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)+'-'+str(time_struct.tm_mday)+'_'+str(time_struct.tm_hour+3)+'-'+str(time_struct.tm_min)+'-'+str(time_struct.tm_sec)
        print("New train:", time_now)
        train_path = os.path.join(PATH_TO_SAVE, time_now)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
    
    model.to(device)  
    #print(model)

    best_acc = 0.0
    saved_epochs = 0

    #wand integration
    if load_from_file:
        #restore wandb
        with open(os.path.join(train_path, "wandb.txt"), 'r') as file1:
            wandb_saved_id = file1.read()

        wandb.init(id=wandb_saved_id, resume=True, project="Mobiledet backbone", name = saved_model, config=wandb_config)

        #restore epochs number
        with open(os.path.join(train_path, "hyp.txt"), 'r') as file2:
            saved_epochs = int(file2.read())
    else:
        wandb_saved_id = wandb.util.generate_id()
        wandb.init(id=wandb_saved_id, resume=True, project="Mobiledet backbone", name = time_now, config=wandb_config)
        
        #save id in the same folder
        with open(os.path.join(train_path, "wandb.txt"), 'w') as file1:
            file1.write(wandb_saved_id)
            file1.close()
        
        #save epoch init number
        with open(os.path.join(train_path, "hyp.txt"), 'w') as file2:
            file2.write('0')
            file2.close()
        
    print("wandb id:", wandb_saved_id)

    if device == torch.device("cuda"):
        print("Use wandb watch to collect model info")
        #wandb.watch(model, log_freq=wandb_log_interval)

    time_beginning = time.time()
    losses = {'train': [], "val": []}
    accuracy = {'train': [], "val": []}
    #log_template = "\nEpoch {ep:03d} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f}"

    #pbar = trange(num_epochs, desc="Epoch")
    for epoch in range(saved_epochs, num_epochs):

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

                    iter += 1

                    #calculate 
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                        processed_data += inputs.size(0)

                        running_accuracy=(running_corrects/(batch_size*iter))

                        if iter % wandb_log_interval == 0:
                            wandb.log({"train": {"loss": loss.item(), "accuracy": running_accuracy.item(), "lr": scheduler.get_last_lr()[0]}})

                        #if iter % 100 == 0:
                        #    #torch.save(model.state_dict(), os.path.join(PATH_TO_SAVE, "tmp.pt"))
                        #    torch.save(model.state_dict(), os.path.join(train_path, "last.pt"))

                        tepoch.set_postfix(loss=loss.item(), accuracy=running_accuracy.item())
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
                wandb.log({"train": {"epoch loss": epoch_loss, "epoch accuracy": epoch_acc}})
                
            if phase == 'val':
                with open(os.path.join(train_path, "hyp.txt"), 'w') as f:
                    f.write(epoch+1)
                    f.close()
                correct_top1 = correct_top1.item() / processed_data
                correct_top5 = correct_top5.item() / processed_data
                torch.save(model.state_dict(), os.path.join(train_path, "last.pt"))
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(train_path, "best.pt"))
                wandb.log({"val": {"loss": loss, "top1 accuracy": correct_top1, "top5 accuracy": correct_top5}})

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
        with open("cuda_datapath.txt", 'r') as f:
            imagenet_dir = f.read()
            print("cuda imagenet path:", imagenet_dir)
    else:
        if torch.backends.mps.is_available():
            print('GPU on M1 MAC is available!  Training on GPU ...')
            train_on_gpu = True
            device = torch.device("mps")
            with open("mps_datapath.txt", 'r') as f:
                imagenet_dir = f.read()
                print("mps imagenet path:", imagenet_dir)
        else:
            print('CUDA and MPS are not available.  Training on CPU ...')
            with open("cuda_datapath.txt", 'r') as f:
                imagenet_dir = f.read()
                print("cuda imagenet path:", imagenet_dir)


    #Loading dataset - like Imagenet, where 1k folders with each classes
    data_transforms = transforms.Compose([
        transforms.Resize((input_size,input_size)),
        transforms.CenterCrop(CenterCrop),
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

    if load_from_file:
        #model.load_state_dict(saved_model)
        model.load_state_dict(torch.load(os.path.join(os.path.join(PATH_TO_SAVE, saved_model), "last.pt"))) #./wandb/run-20221106_155148-2s2e31rf

    #model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
    #model.classifier = nn.Linear(1280, n_classes)

    #The optimal learning rate of the MobileNet-V2 model is 1.66 × 10 −3 - ???
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), init_lr, momentum=init_momentum, weight_decay=init_weight_decay) # CHANGE WANDB!!!
    loss_fn = nn.CrossEntropyLoss()
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(len(imagenet_data['train'])/batch_size))

    train(model, data_loaders, loss_fn, optimizer, exp_lr_scheduler, EPOCHS)