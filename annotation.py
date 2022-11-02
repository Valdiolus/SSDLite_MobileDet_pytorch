import fiftyone as fo
from PIL import Image
import os, glob
import shutil

CLASSES = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]

CLASSES_DICTIONARY = {"person":0, "car":1, "bicycle":2, "motorcycle":3, "bus":4, "truck":5}

DATASET_DATA_DIRS = ["/Users/valdis/datasets/coco6/val", "/Users/valdis/datasets/coco6/train"]#["/media/valdis/NVME/datasets/VOC2012/train", "/media/valdis/NVME/datasets/coco6/train", "/media/valdis/NVME/datasets/bdd100k/train", "/media/valdis/NVME/datasets/rwc/rwc_combined/first_05_02/train"]#["/media/valdis/MD/datasets/VOC2012/train", "/media/valdis/MD/datasets/coco6/train", "/media/valdis/MD/datasets/bdd100k/train", "/media/valdis/MD/datasets/rwc/rwc_combined/first_05_02/train"] # "/media/valdis/MD/datasets/VOC2012/train", "/media/valdis/MD/datasets/coco6/train", "/media/valdis/MD/datasets/bdd100k/train", 
DATASET_LABELS_DIRS = ["/Users/valdis/datasets/coco6/val/labels_filtered.json", "/Users/valdis/datasets/coco6/train/labels_filtered.json"]#["/media/valdis/NVME/datasets/VOC2012/train/labels_filtered.json", "/media/valdis/NVME/datasets/coco6/train/labels_filtered.json","/media/valdis/NVME/datasets/bdd100k/train/labels_filtered.json", "/media/valdis/NVME/datasets/rwc/rwc_combined/first_05_02/train/labels_filtered.json"] # "/media/valdis/MD/datasets/VOC2012/train/labels_filtered.json", "/media/valdis/MD/datasets/coco6/train/labels_filtered.json", "/media/valdis/MD/datasets/bdd100k/train/labels_filtered.json", 

#DATASET_TYPE = "train"

#class x0 y0 width height

# rm -r /media/valdis/MD/datasets/VOC2012/train/data_mdet /media/valdis/MD/datasets/coco6/train/data_mdet /media/valdis/MD/datasets/bdd100k/train/data_mdet /media/valdis/MD/datasets/rwc/rwc_combined/first_05_02/train/data_mdet

MAKE_DATA_COPY = 1

def from_coco():

    for dat_numb in range(len(DATASET_DATA_DIRS)):
        print(DATASET_DATA_DIRS[dat_numb])
        print(DATASET_LABELS_DIRS[dat_numb])

        dataset = fo.Dataset.from_dir(
        data_path=DATASET_DATA_DIRS[dat_numb] + "/data",
        labels_path=DATASET_LABELS_DIRS[dat_numb],
        dataset_type=fo.types.COCODetectionDataset)

        

        new_data_dir_path = DATASET_DATA_DIRS[dat_numb] + "/data_mdet"

        yolov5_data_path = DATASET_DATA_DIRS[dat_numb] + "/data_mdet/images"

        if MAKE_DATA_COPY:
            #create folder for images and labels 
            if not os.path.exists(new_data_dir_path):
                os.makedirs(new_data_dir_path) 

            #create folder for images
            if not os.path.exists(yolov5_data_path):
                os.makedirs(yolov5_data_path)  
            
            new_labels_dir_path = new_data_dir_path + "/labels"
        else:
            new_labels_dir_path = DATASET_DATA_DIRS[dat_numb] + "/labels"

        

        if not os.path.exists(new_labels_dir_path):
            os.makedirs(new_labels_dir_path)          

        without_labels = 0
        for sample in dataset:

            if sample.ground_truth.detections: # check if detections are there
            
                img = Image.open(sample.filepath)
            
                path = new_labels_dir_path + "/" + os.path.splitext(os.path.basename(sample.filepath))[0] + ".txt"

                if MAKE_DATA_COPY:
                    shutil.copyfile(sample.filepath, yolov5_data_path + "/" + os.path.splitext(os.path.basename(sample.filepath))[0] + ".jpg")

                #print(path)
                file = open(path, 'w')

                for detection in sample.ground_truth.detections:
                    det_bbox =  detection.bounding_box
                    bbox_width = det_bbox[2]
                    bbox_height = det_bbox[3]
                    bbox_x_center = det_bbox[0] + bbox_width/2
                    bbox_y_center = det_bbox[1] + bbox_height/2
                    file.write("{} {} {} {} {}\n".format(CLASSES_DICTIONARY[detection.label], bbox_x_center, bbox_y_center, bbox_width, bbox_height))
                    #print(detection.label, detection.bounding_box)
                
                file.close()
            else:
                without_labels += 1
        if not MAKE_DATA_COPY:
            os.symlink(DATASET_DATA_DIRS[dat_numb] + "/data", DATASET_DATA_DIRS[dat_numb] + "/images")
        
        print("Removed with no labels:", without_labels)

from_coco()