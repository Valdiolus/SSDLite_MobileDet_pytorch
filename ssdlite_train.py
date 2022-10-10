import os
from tqdm.autonotebook import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torchvision
from torchvision import datasets, models
import transforms #additional local file
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import mobiledet

from torchsummary import summary
from sklearn.preprocessing import LabelEncoder
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from pathlib import Path
import math
import cv2


# CUDA_VISIBLE_DEVICES="" for CPU run

imagenet_dir = '/media/valdis/NVME/datasets/imagenet/ILSVRC/Data/CLS-LOC'
detector_train_dir = '/media/valdis/NVME/datasets/coco6'
DATA_MODES = ['train', 'val']
batch_size = 128
workers = 4
EPOCHS = 100
n_classes = 6
input_size = 320
PATH_TO_SAVE = './runs'

#3090 2.4  it/s 128bs 8 workers coco only
#3090 2.41 it/s 128bs 4 workers coco only (+- same, 0.01 better)

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

    best_loss = 100.0

    time_beginning = time.time()
    losses = {'train': [], "val": []}
    losses_reg = {'train': [], "val": []}
    losses_class = {'train': [], "val": []}
    #log_template = "\nEpoch {ep:03d} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f}"

    #pbar = trange(num_epochs, desc="Epoch")
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()

            running_loss = 0.0
            running_loss_reg = 0.0
            running_loss_class = 0.0
            #for inputs, labels in dataloader:
            with tqdm(dataloaders[phase], leave=False, desc=f"{phase} iter") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}, {phase} iter")
                for data in tepoch:
                    inputs, labels = data

                    if train_on_gpu:
                        inputs = (inputs/255).to(device)
                        bbox = labels['boxes'].to(device)
                        classes = labels['labels'].to(device)
                    else:
                        inputs = (inputs/255)
    
                    if phase == "train":
                        optimizer.zero_grad()

                    if phase == "train":
                        outputs = model(inputs) #bboxes[4], classes[7] == 11
                    else:
                        with torch.no_grad():
                                outputs = model(inputs)  

                    loss, lb, lc = loss_fn(outputs[:,:,4:], outputs[:,:,:4], classes[:,:,0], bbox)
                    #preds = torch.argmax(outputs, 1)

                    #calculate 
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_loss_reg += lb.item() * inputs.size(0)
                        running_loss_class += lc.item() * inputs.size(0)

                        tepoch.set_postfix(loss=loss.item(), loss_reg=lb.item(), loss_class=lc.item())
                    if phase == "val":
                        running_loss += loss.item() * inputs.size(0)
                        running_loss_reg += lb.item() * inputs.size(0)
                        running_loss_class += lc.item() * inputs.size(0)
                        tepoch.set_postfix(loss=loss.item(), loss_reg=lb.item(), loss_class=lc.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_loss_reg = running_loss_reg / dataset_sizes[phase]
            epoch_loss_class = running_loss_class / dataset_sizes[phase]

            losses[phase].append(epoch_loss)
            losses_reg[phase].append(epoch_loss_reg)
            losses_class[phase].append(epoch_loss_class)

            if phase == 'train':
                scheduler.step()
                
            if phase == 'val':
                torch.save(model.state_dict(), os.path.join(train_path, "last.pt"))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), os.path.join(train_path, "best.pt"))

        #Print log each epoch
        print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train loss", "{:.4f}".format(losses['train'][epoch]), "train loss bbox", "{:.4f}".format(losses_reg['train'][epoch]), "train loss class", "{:.4f}".format(losses_class['train'][epoch]), 
                                                       "val loss", "{:.4f}".format(losses['val'][epoch]), "train loss bbox", "{:.4f}".format(losses_reg['val'][epoch]), "train loss class", "{:.4f}".format(losses_class['val'][epoch]))
    
    time_elapsed = time.time() - time_beginning
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    #print("Final top5:", correct_top5, "top1:", correct_top1)

class DetectionDataset(Dataset):
    def __init__(self, filepath, mode, transforms, max_objs):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.max_objs = max_objs

        images = list(Path(os.path.join(os.path.join(filepath, mode), 'data_mdet/images')).rglob('*.jpg'))

        self.images = sorted(images)

        self.len_ = len(self.images)
        
        labels = list(Path(os.path.join(os.path.join(filepath, mode), 'data_mdet/labels')).rglob('*.txt'))
        #class 0 - background, bbox: Xc,Yc,W,H

        self.labels = sorted(labels)

        print(self.mode, "part with len:", self.len_)

        #print("len labels:", len(self.labels))

        #print(self._prepare_labels(self.labels[0]))
                      
    def __len__(self):
        return self.len_

    def letterbox_resize(self, im, boxes):
        old_size = im.size  # old_size[0] is in (width, height) format
        #print("old size", old_size)#640, 480

        ratio = float(input_size)/max(old_size)
        #print("ratio", ratio) #0.5
        new_size = tuple([int(x*ratio) for x in old_size])
        #print(new_size) #320, 240
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        imn = im.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (input_size, input_size))
        new_im.paste(imn, ((input_size-new_size[0])//2,
                            (input_size-new_size[1])//2))

        x_ratio = ((input_size-new_size[0])/2)/new_size[0]
        y_ratio = ((input_size-new_size[1])/2)/new_size[1]
        #print("x ratio", x_ratio, "y ration", y_ratio)

        #draw = ImageDraw.Draw(new_im)
        for i in range(self.max_objs):
            if boxes[i][0] == boxes[i][1] == boxes[i][2] == boxes[i][3] == 0:
                break
            boxes[i][0] += x_ratio 
            boxes[i][1] += y_ratio 
            #draw.rectangle(((boxes[i][0]*new_size[0], boxes[i][1]*new_size[1]), ((boxes[i][0]+boxes[i][2])*new_size[0], (boxes[i][1]+boxes[i][3])*new_size[1])),outline="red")
        #new_im.save("img.jpg")

        #to tensor only after resizeing bbox
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return new_im, boxes
      
    def _prepare_sample(self, file):
        image = Image.open(file)#.convert("RGB") ???
        #if image.size[0] < input_size or image.size[1] < input_size:
            #print(image.size)
        #image.load()
        #image = image.resize((input_size, input_size))
        return image#np.array(image)
    
    def _prepare_labels(self, file):
        boxes = [[0 for x in range(4)] for y in range(self.max_objs)]# 4 bboxes
        classes = [[0] for y in range(self.max_objs)]# 7 classes
        num_obj = 0
        with open(file) as file1:
            for line in file1:
                #print(np.asarray(line.strip().split(" "), dtype=float))
                arr = line.strip().split(" ")#np.array(line.strip().split(" "), dtype=np.float)
                boxes[num_obj] = [float(x) for x in arr[1:]]
                classes[num_obj] = [int(arr[0])+1]# 0 - background, all classes+1
                #classes[num_obj][int(arr[0])+1] = 1
                num_obj+=1
                if num_obj >= self.max_objs:
                    break
        #print(boxes, classes)
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        return boxes, classes #np.array(labels)#.astype(np.float)
  
    def __getitem__(self, index):
        img = self._prepare_sample(self.images[index])
        #img = torch.tensor(img / 255, dtype=torch.float32)
        #x = self.transform(x)

        boxes, labels = self._prepare_labels(self.labels[index])

        image_id = torch.tensor([index])

        img, boxes = self.letterbox_resize(img, boxes)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id

        

        if self.transforms is not None:
            x, y = self.transforms(img, target)
        else:
            print("transforms error")
        return x, y

class MultiboxLoss(nn.Module):
    def __init__(self, iou_threshold, neg_pos_ratio,
                 center_variance, size_variance, device):
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        #self.priors = priors
        #self.priors.to(device)

    def hard_negative_mining(self, loss, labels, neg_pos_ratio):
        """
        It used to suppress the presence of a large number of negative prediction.
        It works on image level not batch level.
        For any example/image, it keeps all the positive predictions and
        cut the number of negative predictions to make sure the ratio
        between the negative examples and positive examples is no more
        the given ratio for an image.
        Args:
            loss (N, num_priors): the loss for each example.
            labels (N, num_priors): the labels.
            neg_pos_ratio:  the ratio between the negative examples and positive examples.
        """
        pos_mask = labels > 0
        num_pos = pos_mask.long().sum(dim=1, keepdim=True)
        num_neg = num_pos * neg_pos_ratio

        loss[pos_mask] = -math.inf
        _, indexes = loss.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_mask = orders < num_neg
        return pos_mask | neg_mask

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        #print(confidence.shape, predicted_locations.shape, labels.shape, gt_locations.shape)
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = self.hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        lb = smooth_l1_loss/num_pos
        lc = classification_loss/num_pos
        return lb+lc, lb, lc #smooth_l1_loss/num_pos + classification_loss/num_pos

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

data_transforms = transforms.Compose([
    #transforms.ToTensor(),
    transforms.PILToTensor(),
    #transforms.ConvertImageDtype(torch.float),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
])
dataset = {x: DetectionDataset(detector_train_dir, mode=x, transforms=data_transforms, max_objs=2034) 
                for x in ['train', 'val']}
data_loaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batch_size, shuffle=True, num_workers=workers)
                for x in ['train', 'val']}

dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}
print(dataset_sizes)
#for _ in range(1000):
#    images, targets = next(iter(data_loaders['val']))
#print(images.shape, targets['boxes'].shape, targets['labels'].shape)
#print(images[0], targets['labels'])

#exit(0)
#Configurate model and hyperparameters

model = mobiledet.MobileDetTPU(net_type="detector", classes=n_classes)

#model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
#model.classifier = nn.Linear(1280, n_classes)

#summary(model.to(device), (3, input_size, input_size))
#print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = MultiboxLoss(iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=device)#losses#nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train(model, data_loaders, loss_fn, optimizer, exp_lr_scheduler, EPOCHS)