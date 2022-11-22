import os
from tqdm.autonotebook import tqdm, trange
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import itertools

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
from prettytable import PrettyTable

#for time correction
gmt_dst = 2

#use fp16
mixed_precision=True

# CUDA_VISIBLE_DEVICES="" for CPU run

detector_train_dir = '/media/valdis/NVME/datasets/coco6'
validation_fo_file = '/media/valdis/NVME/datasets/coco6/val/labels_filtered.json' # same dir - /data with images
DATA_MODES = ['train', 'val']
batch_size = 256
workers = 4
EPOCHS = 100
n_classes = 6
CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
input_size = 320
PATH_TO_SAVE = './runs'

init_lr = 0.13
init_momentum = 0.9
init_weight_decay = 0.00004

anchor_boxes = 0
varx = 0.1
vary = 0.1
varw = 0.2
varh = 0.2

max_pred_per_image = 100

SHOW_RESULTS_EACH_EPOCH = 0

#fine-tune controls
use_pretrain = 0
use_backbone = 1
freeze_backbone = 0

backbone_saves = 'backbone_2022-11-16_9-56-38'
pretrained_saves = ''






#3090 2.4  it/s 128bs 8 workers coco only fp32
#3090 2.41 it/s 128bs 4 workers coco only fp32 (+- same, 0.01 better)

#mac m1 pro 16gpu 2.1s/it 64bs 4 workers coco only (1/10 of 3090!!!)

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2]) *
                              (boxes[:, 3]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    #print("areas", areas1, areas2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2]-boxes1[:, None, 2:]/2, boxes2[:, :2]-boxes2[:, 2:]/2)
    inter_lowerrights = torch.min(boxes1[:, None, :2]+boxes1[:, None, 2:]/2, boxes2[:, :2]+boxes2[:, 2:]/2)
    #print("inter_upper", inter_upperlefts)
    #print("inter_lower", inter_lowerrights)
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def box_iou_nms(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[2]) *
                              (boxes[3]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    #print("areas", areas1, areas2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:2]-boxes1[2:]/2, boxes2[:2]-boxes2[2:]/2)
    inter_lowerrights = torch.min(boxes1[:2]+boxes1[2:]/2, boxes2[:2]+boxes2[2:]/2)
    #print("inter_upper", inter_upperlefts)
    #print("inter_lower", inter_lowerrights)
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    #print(inters)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[0] * inters[1]
    union_areas = areas1 + areas2 - inter_areas
    return inter_areas / union_areas

def generate_anchor_boxes(MinScale=.2, MaxScale=0.95, asp=[1.0, 2.0, 0.5, 3.0, 0.33, 0.66]):

    #ssd mnet example - numBoxes=[4,6,6,6,4,4], layerWidth=[28,14,7,4,2,1], k = 10+1+4):

    #iou = box_iou(torch.as_tensor([[10,20,200,200], [20,30,120,120]]), torch.as_tensor([[90,80,160,150], [150,120,190,250]]))
    #print("iou", iou.shape, iou)

    numBoxes=[3,6,6,6,6,6]
    layerWidths=[20,10,5,3,2,1]

    #MinScale = .2 # Min and Max scale given as percentage
    #MaxScale = 0.95
    scales = [ MinScale + x/len(layerWidths) * (MaxScale-MinScale) for x in range(len(layerWidths)) ]
    scales = scales[::-1] # reversing the order because the layerWidths go from high to low (lower to higher resoltuion)

    #asp = [1.0, 2.0, 0.5, 3.0, 0.33, 0.66]#need 6!!! - check 0.66!!!
    asp1 = [x**0.5 for x in asp]
    asp2 = [1/x for x in asp1]

    BOXES = sum([a*a*b for a,b in zip(layerWidths,numBoxes)])
    centres = np.zeros((BOXES,2))
    hw = np.zeros((BOXES,2))
    boxes = np.zeros((BOXES,4))

    idx = 0

    for gridSize, numBox, scale in zip(layerWidths,numBoxes,scales):
        step_size = input_size*1.0/gridSize

        for i in range(gridSize):
            for j in range(gridSize):
                pos = idx + (i*gridSize+j) * numBox
                centres[ pos : pos + numBox , :] = i*step_size + step_size/2, j*step_size + step_size/2
                hw[ pos : pos + numBox , :] = np.multiply(gridSize*scale, np.squeeze(np.dstack([asp1,asp2]),axis=0))[:numBox,:]

        idx += gridSize*gridSize*numBox 
    

    boxes[:,0] = centres[:,0]/input_size
    boxes[:,1] = centres[:,1]/input_size
    boxes[:,2] = hw[:,0]/input_size
    boxes[:,3] = hw[:,1]/input_size
    boxes = torch.as_tensor(boxes, dtype=torch.float32)

    return boxes

    def bestIoU(searchBox):
        best = box_iou(searchBox, boxes)
        print(best.shape, best)
        return torch.argwhere(best > 0.5)
    
    #corner = np.random.randint(320 - 28, size=(600,2))
    #print(corner.shape, corner) #(600, 2)

    images, targets = next(iter(data_loaders['train']))
    for i in range(images.shape[0]):
        n = 0
        for j in range(2034):
            if targets['boxes'][i][j][0] == targets['boxes'][i][j][1] == targets['boxes'][i][j][2] == targets['boxes'][i][j][3]:
                break
            n += 1
        #print(targets['boxes'][i].shape) # torch.Size([2034, 4])
        print("non-zero:", n)
        time_beginning = time.time()
        box_idx = bestIoU(targets['boxes'][i][:n]).numpy()#.astype(np.uint16)

        print("time", time.time() - time_beginning)
        print("arg:", box_idx)
        for k in range(len(box_idx)):
            print("number:", targets['boxes'][i][box_idx[k][0]], boxes[box_idx[k][1]])
    
def generate_prediction_boxes(predictions, conf_th=0.5, iou_th=0.5):
    time_b = time.time()

    #Xc, Yc, W, H of decoded boxes (2034)

    #Calculate softmax across all preds, find max class in each prediction, find which predictions have > threshold
    output_conf_full = nn.Softmax(dim=2)(predictions[:,:,4:])
    output_labels_full = torch.argmax(output_conf_full, 2)
    output_conf = torch.argwhere(output_conf_full > conf_th)#[1,1280,3] - number of batch, pred and in class

    results = np.zeros((batch_size, max_pred_per_image, 6))
    #decode bboxes 
    for b in range(batch_size):
        filtered_output = torch.zeros((2034, 6)) # max predictions per image + image ID
        n_pred_per_image = 0
        no_preds = True
        for i in range(output_conf.shape[0]):
            if output_conf[i][0] != b:
                continue
            no_preds = False
            #REMOVE 0 LABELS???
            filtered_output[n_pred_per_image][0] = predictions[output_conf[i][0]][output_conf[i][1]][0].item() * varx * anchor_boxes[output_conf[i][1]][2] + anchor_boxes[output_conf[i][1]][0]
            filtered_output[n_pred_per_image][1] = predictions[output_conf[i][0]][output_conf[i][1]][1].item() * vary * anchor_boxes[output_conf[i][1]][3] + anchor_boxes[output_conf[i][1]][1]
            filtered_output[n_pred_per_image][2] = np.exp(predictions[output_conf[i][0]][output_conf[i][1]][2].item() * varw) * anchor_boxes[output_conf[i][1]][2]
            filtered_output[n_pred_per_image][3] = np.exp(predictions[output_conf[i][0]][output_conf[i][1]][3].item() * varh) * anchor_boxes[output_conf[i][1]][3]
            #conf
            filtered_output[n_pred_per_image][4] = output_conf_full[output_conf[i][0]][output_conf[i][1]][output_labels_full[output_conf[i][0]][output_conf[i][1]]].item()
            #label
            filtered_output[n_pred_per_image][5] = output_labels_full[output_conf[i][0]][output_conf[i][1]].item() - 1 # -1 - remove background [0]
            n_pred_per_image += 1
        
        if no_preds:
            continue
        
        #apply NMS and filter only best predictions across each 
        nms_boxes = []
        for i in range(n_pred_per_image):
            not_save = False
            for j in range(n_pred_per_image):
                #skip same boxes
                if i == j:
                    continue
                #skip different classes - 100% not the same boxes to compare
                if filtered_output[j][5] != filtered_output[i][5]:
                    continue
                #apply iou
                iou = box_iou_nms(filtered_output[i][:4], filtered_output[j][:4])
                if iou > iou_th:
                    if filtered_output[j][4] > filtered_output[i][4]:
                        not_save = True

            if not not_save:
                #filtered_output[i][6] = image_ids[b].item()
                nms_boxes.append([filtered_output[i][x] for x in range(6)])

        #find only top k results on the image
        nms_boxes = np.array(nms_boxes)
        topk = np.argsort(nms_boxes[:,4], axis=-1)
        topk = topk[::-1][:max_pred_per_image]

        #add to results in the given image number
        results[b][:nms_boxes[topk].shape[0]] = nms_boxes[topk]
    
    #print("post process duration:", time.time() - time_b)
    return results

def validate_batch(tp_table, labels, results, iou_th=0.5):
    tp_table = np.zeros((n_classes, 6))

    i_groudtruth = np.zeros(n_classes)
    i_predictions = np.zeros(n_classes)
    i_truepos = np.zeros(n_classes)
    for b in range(labels['real_data'].shape[0]):
        for gt_preds in range(labels['real_data'].shape[1]):

            if labels['real_data'][b][gt_preds][2] == 0 and labels['real_data'][b][gt_preds][3] == 0:
                #no width and height of bbox -> next images
                break
            
            gt_label = int(labels['real_data'][b][gt_preds][4])
            gt_box = labels['real_data'][b][gt_preds][0:4]
            i_groudtruth[gt_label] += 1
            i_predictions[gt_label] = 0
            iou = []

            for res_preds in range(results.shape[1]):
                if results[b][res_preds][2] == 0 and results[b][res_preds][3] == 0:
                    #no width and height of bbox -> view iou and next gt
                    break
                if gt_label == results[b][res_preds][5]:
                    i_predictions[gt_label] += 1
                    iou_ = box_iou_nms(gt_box, torch.as_tensor(results[b][res_preds][0:4])).item()
                    iou.append(iou_)
            
            if len(iou) > 0:
                if max(iou) > iou_th:
                    i_truepos[gt_label] += 1
                    if i_truepos[gt_label] > i_predictions[gt_label]:
                        i_truepos[gt_label] = i_predictions[gt_label]

        for cl in range(n_classes):
            tp_table[cl, 0] += i_truepos[cl]                    # tp - find best in predictions
            # tp_table[cl, 1] - true negative
            tp_table[cl, 2] += i_predictions[cl]-i_truepos[cl]  # fp - pred boxes not match with gt boxes
            tp_table[cl, 3] += i_groudtruth[cl]-i_truepos[cl]   # fn - gt boxes not match with pred boxes
            tp_table[cl, 4] += i_groudtruth[cl]                 # number of groudn truth boxes
            tp_table[cl, 5] += i_predictions[cl]                # number of predicted boxes
    
    return tp_table

def calculate_map(tp_table):
    #class name, precision, recall, true positive, false positive, false negative, number of ground truth objects, number of predicted objects
    table = PrettyTable(["class", "prec", "rec", "tp", "fp", "fn", "gt", "pred"])

    pr_table = np.zeros((n_classes, 7)) # precision, recall, tp, fp, fn, gt_supp, pred_supp

    calc_AP = 0

    for cls in range(n_classes):
        if not tp_table[cls, 0] == 0 and (tp_table[cls, 2] == 0 or tp_table[cls, 3] == 0):
            pr_table[cls, 0] = tp_table[cls, 0]/(tp_table[cls, 0] + tp_table[cls, 2]) 
            pr_table[cls, 1] = tp_table[cls, 0]/(tp_table[cls, 0] + tp_table[cls, 3]) 
        
        pr_table[cls, 2] = tp_table[cls, 0]
        pr_table[cls, 3] = tp_table[cls, 2]
        pr_table[cls, 4] = tp_table[cls, 3]
        pr_table[cls, 5] = tp_table[cls, 4]
        pr_table[cls, 6] = tp_table[cls, 5]

        calc_AP += pr_table[cls, 0]

    calc_AP = calc_AP / n_classes

    if SHOW_RESULTS_EACH_EPOCH:
        for cls in range(n_classes):
            new_row = [CLASSES[cls]] + ["{:.2f}".format(pr_table[cls, x]) for x in range(2)] + ["{:.0f}".format(pr_table[cls, y]) for y in range(2,7)]
            table.add_row(new_row)

        print("AP:", calc_AP)
        print(table)

    return pr_table, calc_AP


def train(model, dataloaders, loss_fn, optimizer, scheduler, num_epochs = 1):  
    #count the current date and time
    time_struct = time.gmtime()
    time_now = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)+'-'+str(time_struct.tm_mday)+'_'+str(time_struct.tm_hour+gmt_dst)+'-'+str(time_struct.tm_min)+'-'+str(time_struct.tm_sec)
    print(time_now)
    train_path = os.path.join(PATH_TO_SAVE, time_now)
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    
    model.to(device)  
    #print(model)

    best_loss = 100.0
    priors = 1

    time_beginning = time.time()
    losses = {'train': [], "val": []}
    losses_reg = {'train': [], "val": []}
    losses_class = {'train': [], "val": []}
    #log_template = "\nEpoch {ep:03d} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f} train_loss: {t_loss:0.4f} val_loss {v_loss:0.4f}"
    
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == 'train':
                model.train(True)
            else:
                model.eval()

            #validation
            tp_table = np.zeros((n_classes, 6))

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
                        #image_ids = labels['image_id']
                        #objs = labels['objs'].to(device)
                    else:
                        inputs = (inputs/255)
                    
                    if mixed_precision:
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            if phase == "train":
                                outputs = model(inputs) #bboxes[4], classes[7] == 11
                            else:
                                with torch.no_grad():
                                        outputs = model(inputs)  
                                        
                            loss, lb, lc = loss_fn(outputs[:,:,4:], outputs[:,:,:4], classes, bbox)
                    else:
                        if phase == "train":
                            outputs = model(inputs) #bboxes[4], classes[7] == 11
                        else:
                            with torch.no_grad():
                                    outputs = model(inputs)  

                        loss, lb, lc = loss_fn(outputs[:,:,4:], outputs[:,:,:4], classes, bbox)

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

                        scheduler.step()

                        running_loss += loss.item() * inputs.size(0)
                        running_loss_reg += lb.item() * inputs.size(0)
                        running_loss_class += lc.item() * inputs.size(0)

                        tepoch.set_postfix(loss=loss.item(), loss_reg=lb.item(), loss_class=lc.item())
                    if phase == "val":
                        #validation
                        #val_results = generate_prediction_boxes(outputs, conf_th=0.25, iou_th=0.1)
                        #tp_table = validate_batch(tp_table, labels, val_results, iou_th=0.1)

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
                a=1

            if phase == 'val':
                #pr_table, calc_AP = calculate_map(tp_table)
                torch.save(model.state_dict(), os.path.join(train_path, "last.pt"))
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), os.path.join(train_path, "best.pt"))

        #Print log each epoch "| val AP", "{:.2f}".format(calc_AP), 
        print("\nEpoch %s/%s" % (epoch+1, num_epochs), "train loss", "{:.3f}".format(losses['train'][epoch]), "train loss reg", "{:.3f}".format(losses_reg['train'][epoch]), "train loss class", "{:.3f}".format(losses_class['train'][epoch]), 
                                                       "val loss", "{:.3f}".format(losses['val'][epoch]), "val loss reg", "{:.3f}".format(losses_reg['val'][epoch]), "val loss class", "{:.3f}".format(losses_class['val'][epoch]))

    time_elapsed = time.time() - time_beginning
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    #print("Final top5:", correct_top5, "top1:", correct_top1)

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
        #check where to get sigmoid focal loss
        #classification_loss = F.sigmoid_focal_loss(confidence.reshape(-1, num_classes), labels[mask], alpha=0.75, gamma=2.0) 
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        lb = smooth_l1_loss/num_pos
        lc = classification_loss/num_pos
        return lb+lc, lb, lc #smooth_l1_loss/num_pos + classification_loss/num_pos
        
class DetectionDataset(Dataset):
    def __init__(self, filepath, mode, transforms, preds):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.preds = preds

        images = list(Path(os.path.join(os.path.join(filepath, mode), 'data_mdet/images')).rglob('*.jpg'))
        
        self.images = sorted(images)

        if mode == 'val':
            self.len_ = len(self.images)
        else:
            self.len_ = len(self.images)
        
        labels = list(Path(os.path.join(os.path.join(filepath, mode), 'data_mdet/labels')).rglob('*.txt'))
        #class 0 - background, bbox: Xc,Yc,W,H

        self.labels = sorted(labels)

        print(self.mode, "part with len:", self.len_)

        self.anchor_boxes = anchor_boxes

        self.varx = varx
        self.vary = vary
        self.varw = varw
        self.varh = varh

        #print("len labels:", len(self.labels))

        #print(self._prepare_labels(self.labels[0]))
                      
    def __len__(self):
        return self.len_

    def letterbox_resize(self, im):
        old_size = im.size  # old_size[0] is in (width, height) format
        #print("old size", old_size)#640, 480

        #draw = ImageDraw.Draw(im)
        #for i in range(self.max_objs):
        #    if boxes[i][0] == boxes[i][1] == boxes[i][2] == boxes[i][3] == 0:
        #        break
        #    draw.rectangle((((boxes[i][0]-boxes[i][2]/2)*old_size[0], (boxes[i][1]-boxes[i][3]/2)*old_size[1]), 
        #                    ((boxes[i][0]+boxes[i][2]/2)*old_size[0], (boxes[i][1]+boxes[i][3]/2)*old_size[1])),outline="red")
        #im.save("img_old.jpg")

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
        #for i in range(self.max_objs):
        #    if boxes[i][0] == boxes[i][1] == boxes[i][2] == boxes[i][3] == 0:
        #        break
        #    boxes[i][0] += x_ratio 
        #    boxes[i][1] += y_ratio 
            #draw.rectangle((((boxes[i][0]-boxes[i][2]/2)*new_size[0], (boxes[i][1]-boxes[i][3]/2)*new_size[1]), 
            #                ((boxes[i][0]+boxes[i][2]/2)*new_size[0], (boxes[i][1]+boxes[i][3]/2)*new_size[1])),outline="red")
        #new_im.save("img.jpg")

        #to tensor only after resizeing bbox
        #boxes = torch.as_tensor(boxes, dtype=torch.float32)
        return new_im, x_ratio, y_ratio
      
    def _prepare_sample(self, file):
        image = Image.open(file)#.convert("RGB") ???
        #if image.size[0] < input_size or image.size[1] < input_size:
            #print(image.size)
        #image.load()
        #image = image.resize((input_size, input_size))
        return image#np.array(image)
    
    def _prepare_labels(self, file, x_ratio, y_ratio):
        gt_boxes = []# 4 bboxes 
        gt_classes = []# 7 classes # 0 - background, all classes+1
        out_boxes = torch.zeros(self.preds, 4)#[[0 for x in range(4)] for y in range(self.preds)]
        out_classes = torch.zeros((self.preds), dtype=torch.int64)#[[0] for y in range(self.preds)]
        out_gt = torch.zeros(self.preds, 5)
        num_obj = 0
        with open(file) as file1:
            for line in file1:
                arr = line.strip().split(" ")
                box = [float(x) for x in arr[1:]]
                box[0] += x_ratio
                box[1] += y_ratio

                #for training
                gt_boxes.append(box)
                gt_classes.append(int(arr[0])+1)

                #for validation
                out_gt[num_obj][0:4] = torch.as_tensor(box)
                out_gt[num_obj][4] = float(arr[0]) # without background

                #classes[num_obj][int(arr[0])+1] = 1
                num_obj+=1
                #if num_obj >= self.preds:
                #    break
        #calculate acnhors and gt iou best id
        #print("gt boxes:", len(gt_boxes))
        #time_beginning = time.time()
        gt_boxes = torch.as_tensor(gt_boxes, dtype=torch.float32)
        iou_gt_to_def = box_iou(gt_boxes, self.anchor_boxes) #len_gt * len_def_boxes

        n_boxes = 0

        for n in range(len(gt_boxes)): # HOW TO WORK WITH "NEUTRAL" BOXES???
            iou_gt = iou_gt_to_def[n]
            good_box_idx = torch.argwhere(iou_gt > 0.5)
            #print("good box ids", n, len(good_box_idx), good_box_idx)
            #print("train boxes:", len(box_idx))
            #print("time:", time.time() - time_beginning)
            if (len(good_box_idx) > 0): # there are > 0.5 iou - get only them (they are best too)
                #print("good 1:", good_box_idx[0][0])
                for k in range(len(good_box_idx)):
                    anc_k = good_box_idx[k][0]
                    out_boxes[n_boxes][0] = ((gt_boxes[n][0] - self.anchor_boxes[anc_k][0])/self.anchor_boxes[anc_k][2])/np.sqrt(self.varx)
                    out_boxes[n_boxes][1] = ((gt_boxes[n][1] - self.anchor_boxes[anc_k][1])/self.anchor_boxes[anc_k][3])/np.sqrt(self.vary)
                    out_boxes[n_boxes][2] = (torch.log(gt_boxes[n][2]/self.anchor_boxes[anc_k][2]))/np.sqrt(self.varw)
                    out_boxes[n_boxes][3] = (torch.log(gt_boxes[n][3]/self.anchor_boxes[anc_k][3]))/np.sqrt(self.varh)

                    out_classes[n_boxes] = gt_classes[n]

                    n_boxes += 1 

            else: # only <0.5 iou left - get only best    
                iou_max = torch.max(iou_gt)
                if iou_max == 0:
                    continue

                #print("iou max", iou_max)
                def_gt_max = torch.argwhere(iou_gt == iou_max)
                #print("where max:", def_gt_max)
                for k in range(len(def_gt_max)):
                    anc_k = def_gt_max[k][0]
                    out_boxes[n_boxes][0] = ((gt_boxes[n][0] - self.anchor_boxes[anc_k][0])/self.anchor_boxes[anc_k][2])/np.sqrt(self.varx)
                    out_boxes[n_boxes][1] = ((gt_boxes[n][1] - self.anchor_boxes[anc_k][1])/self.anchor_boxes[anc_k][3])/np.sqrt(self.vary)
                    out_boxes[n_boxes][2] = (torch.log(gt_boxes[n][2]/self.anchor_boxes[anc_k][2]))/np.sqrt(self.varw)
                    out_boxes[n_boxes][3] = (torch.log(gt_boxes[n][3]/self.anchor_boxes[anc_k][3]))/np.sqrt(self.varh)

                    out_classes[n_boxes] = gt_classes[n]

                    n_boxes += 1
            
                    

        objs = torch.as_tensor([num_obj], dtype=torch.int64)
        #out_boxes = torch.as_tensor(out_boxes, dtype=torch.float32)
        #out_classes = torch.as_tensor(out_classes, dtype=torch.int64)
        return out_boxes, out_classes, objs, out_gt#np.array(labels)#.astype(np.float)
  
    def __getitem__(self, index):
        img = self._prepare_sample(self.images[index])
        #img = torch.tensor(img / 255, dtype=torch.float32)
        #x = self.transform(x)

        img, x_ratio, y_ratio = self.letterbox_resize(img)

        boxes, labels, objs, real_data = self._prepare_labels(self.labels[index], x_ratio, y_ratio)

        image_id = torch.tensor([index])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id # number 1...batch size
        target["objs"] = objs # number of gt objects on the image
        target["real_data"] = real_data

        if self.transforms is not None:
            x, y = self.transforms(img, target)
        else:
            print("transforms error")
        return x, y



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

    anchor_boxes = generate_anchor_boxes()

    data_transforms = transforms.Compose([
        #transforms.ToTensor(),
        transforms.PILToTensor(),
        #transforms.ConvertImageDtype(torch.float),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    dataset = {x: DetectionDataset(detector_train_dir, mode=x, transforms=data_transforms, preds=2034) 
                    for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(dataset[x], batch_size=batch_size, shuffle=True, num_workers=workers)
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(dataset[x]) for x in ['train', 'val']}

    iter_per_epoch = int(len(dataset['train'])/batch_size)

    print(dataset_sizes)

    model = mobiledet.MobileDetTPU(net_type="detector", classes=n_classes)

    #model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2)
    #model.classifier = nn.Linear(1280, n_classes)

    #summary(model.to(device), (3, input_size, input_size))

    if use_pretrain:
        model.load_state_dict(torch.load(os.path.join(os.path.join(PATH_TO_SAVE, pretrained_saves), "last.pt")))
        print("Pretrained SSD model is loaded")

    if use_backbone:
        ssd_dict = model.state_dict()

        backbone_model = mobiledet.MobileDetTPU(net_type="classifier", classes=1000)
        backbone_model.load_state_dict(torch.load(os.path.join(os.path.join(PATH_TO_SAVE, backbone_saves), "last.pt"), map_location=torch.device('cpu')))
        backbone_dict = backbone_model.state_dict()

        for k, v in backbone_dict.items():
            if 'inv12' in k:
                break
            ssd_dict.update({k: v})
        model.load_state_dict(ssd_dict)
        print("Pretrained backbone is loaded")
    
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'inv12' in name:
                freeze_backbone = 0
            
            if freeze_backbone:
                param.requires_grad = False
            else:
                param.requires_grad = True
            #print(name, 0 if param.requires_grad == False else 1)
        print("Backbone is freezed")


    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), init_lr, momentum=init_momentum, weight_decay=init_weight_decay)
    loss_fn = MultiboxLoss(iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2, device=device)
    exp_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(EPOCHS*iter_per_epoch))#


    train(model, data_loaders, loss_fn, optimizer, exp_lr_scheduler, EPOCHS)