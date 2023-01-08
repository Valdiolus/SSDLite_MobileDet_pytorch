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

import argparse

# CUDA_VISIBLE_DEVICES="" for CPU run

#use fp16
mixed_precision=True

#for time correction
gmt_dst = 2

batch_size = 256 if mixed_precision else 128
workers = 8
EPOCHS = 250
n_classes = 1000
input_size = 320
resize = int(input_size/0.875)
init_lr = 0.05
init_momentum = 0.9
init_weight_decay = 0.00004
CenterCrop = input_size # 224? was 298 

load_from_file = 0
saved_model = ''
wandb_saved_id = 0

PATH_TO_SAVE = './runs'

wandb_log_interval = 5 if mixed_precision else 10
wandb_config = {"batch_size": batch_size,
                "num_workers": workers,
                "input size": input_size,
                "resize": resize,
                "epochs": EPOCHS,
                "pin_memory": False,  
                "precision": 16 if mixed_precision else 32,
                "optimizer": "SGD",
                "lr": init_lr,
                "momentum": init_momentum,
                "weight_decay": init_weight_decay,
                "CenterCrop": CenterCrop,
                "aug": "random resize, hor flip"
                }


def train(model, dataloaders, loss_fn, optimizer, scheduler, iter_per_epoch, num_epochs = 10):  
    if load_from_file:
        print("Resume training:", saved_model)
        train_path = os.path.join(PATH_TO_SAVE, saved_model)
    else:
        #count the current date and time
        time_struct = time.gmtime()
        time_now = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)+'-'+str(time_struct.tm_mday)+'_'+str(time_struct.tm_hour+gmt_dst)+'-'+str(time_struct.tm_min)+'-'+str(time_struct.tm_sec)
        print("New train:", time_now)
        train_path = os.path.join(PATH_TO_SAVE, time_now)
        if not os.path.exists(train_path):
            os.makedirs(train_path)
    
    model.to(device)  
    #print(model)

    best_top1 = 0.0
    best_top5 = 0.0
    saved_epochs = 0

    #restore wandb, epochs count and lr scheduler
    if load_from_file:
        #restore wandb
        with open(os.path.join(train_path, "wandb.txt"), 'r') as file1:
            wandb_saved_id = file1.read()

        wandb.init(id=wandb_saved_id, resume=True, project="Mobiledet backbone", name = saved_model, config=wandb_config)

        #restore epochs number
        with open(os.path.join(train_path, "hyp.txt"), 'r') as file2:
            saved_epochs = int(file2.read())

        #restore lr scheduler state
        for _ in range((saved_epochs+1)*iter_per_epoch):
            scheduler.step()
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

    time_beginning = time.time()
    #log_template = "\nEpoch {ep:03d} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f}"

    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    #pbar = trange(num_epochs, desc="Epoch")
    for epoch in range(saved_epochs, num_epochs):
        epoch_losses_train = 0
        epoch_losses_val = 0
        epoch_accuracy_train = 0
        epoch_accuracy_val = 0

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

                    if mixed_precision:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            if phase == "train":
                                outputs = model(inputs)
                            else:
                                with torch.no_grad():
                                        outputs = model(inputs)                   
                        
                            loss = loss_fn(outputs, labels)
                    else:
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
                        optimizer.zero_grad()
                        if mixed_precision:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
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

                        scheduler.step()

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
                        running_corrects += iter_top1

                        tepoch.set_postfix(top1=(correct_top1 / processed_data).item(), top5=(correct_top5 / processed_data).item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                epoch_losses_train = epoch_loss
                epoch_accuracy_train = epoch_acc
                wandb.log({"train": {"epoch loss": epoch_loss, "epoch accuracy": epoch_acc}})
                
            if phase == 'val':
                epoch_losses_val = epoch_loss

                with open(os.path.join(train_path, "hyp.txt"), 'w') as f:
                    f.write(str(epoch+1)) # +1 because of new epoch - current is done 
                    f.close()
                correct_top1 = correct_top1.item() / processed_data
                correct_top5 = correct_top5.item() / processed_data

                torch.save(model.state_dict(), os.path.join(train_path, "last.pt"))

                if correct_top1 > best_top1:
                    best_top1 = correct_top1
                    torch.save(model.state_dict(), os.path.join(train_path, "best_top1.pt"))

                if correct_top5 > best_top5:
                    best_top5 = correct_top5
                    torch.save(model.state_dict(), os.path.join(train_path, "best_top5.pt"))

                wandb.log({"val": {"loss": loss, "top1 accuracy": correct_top1, "top5 accuracy": correct_top5}})

        #Print log each epoch
        print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train acc", "{:.4f}".format(epoch_accuracy_train), "train loss", "{:.4f}".format(epoch_losses_train), 
                                                       "val loss", "{:.4f}".format(epoch_losses_val), "val top1:", correct_top1, "val top5:", correct_top5)
    
    time_elapsed = time.time() - time_beginning
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_top1))
    print("Final top1:", correct_top1, "top5:", correct_top5)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Mobiledet Backbone training")
    #parser.add_argument("-r", "--restore", action="store_true", help="flag to restore")
    parser.add_argument("-r", "--resume", required=False, help="run id to resume")
    args, unknown = parser.parse_known_args()

    saved_model = args.resume
    if saved_model is not None:
        load_from_file = True

    #Check if GPU is enable
    if torch.cuda.is_available():
        print('CUDA is available!  Training on GPU ...')
        train_on_gpu = True
        device = torch.device("cuda")
        with open("cuda_datapath_imagenet.txt", 'r') as f:
            imagenet_dir = f.read()
            print("cuda imagenet path:", imagenet_dir)
    else:
        if torch.backends.mps.is_available():
            print('GPU on M1 MAC is available!  Training on GPU ...')
            train_on_gpu = True
            device = torch.device("mps")
            with open("mps_datapath_imagenet.txt", 'r') as f:
                imagenet_dir = f.read()
                print("mps imagenet path:", imagenet_dir)
        else:
            print('CUDA and MPS are not available.  Training on CPU ...')
            with open("cuda_datapath_imagenet.txt", 'r') as f:
                imagenet_dir = f.read()
                print("cuda imagenet path:", imagenet_dir)


    #Loading dataset - like Imagenet, where 1k folders with each classes
    data_transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(20),
        transforms.ToTensor(),
        ])
    data_transforms_val = transforms.Compose([
        transforms.Resize((resize)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        ])
    imagenet_data = {x: torchvision.datasets.ImageFolder(os.path.join(imagenet_dir, x), transform=data_transforms_train if x=='train' else data_transforms_val)
                    for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(imagenet_data[x], batch_size=batch_size, shuffle=True, num_workers=workers)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(imagenet_data[x]) for x in ['train', 'val']}
    class_names = imagenet_data['train'].classes
    iter_per_epoch = int(len(imagenet_data['train'])/batch_size)
    print(dataset_sizes)


    #Configurate model and hyperparameters
    model = mobiledet.MobileDetTPU(net_type="classifier", classes=n_classes)

    if load_from_file:
        model.load_state_dict(torch.load(os.path.join(os.path.join(PATH_TO_SAVE, saved_model), "last.pt")))

    #The optimal learning rate of the MobileNet-V2 model is 1.66 × 10 −3 - ???
    optimizer = torch.optim.SGD(model.parameters(), init_lr, momentum=init_momentum, weight_decay=init_weight_decay) # CHANGE WANDB!!!
    loss_fn = nn.CrossEntropyLoss()
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(EPOCHS*iter_per_epoch))#

    train(model, data_loaders, loss_fn, optimizer, exp_lr_scheduler, iter_per_epoch, EPOCHS)