#!/usr/bin/env python3

from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

from utilsDetectron import *


# PLEASE FILL THIS INFORMATION REGARDING THE DATASET
dt_name = "Bags"
dt_location = "(local, rodrigo laptop), /home/rodrigo/Development/datasets/bags"
dt_total_num_imgs = 206
dt_classes = ["bag", "bagHandle"]

dt_train_dir = os.getcwd() + "/dataset/train"
dt_train_imgs = int(len([entry for entry in os.listdir(dt_train_dir) if os.path.isfile(os.path.join(dt_train_dir, entry))]) / 2)

dt_test_dir = os.getcwd() + "/dataset/test"
dt_test_imgs = int(len([entry for entry in os.listdir(dt_test_dir) if os.path.isfile(os.path.join(dt_test_dir, entry))]) / 2)

dt_used_imgs = dt_train_imgs + dt_test_imgs

# WRITE INFORMATION TO FILE
dt_info_file = "trainingInfo.txt"

if os.path.exists(dt_info_file):
  os.remove(dt_info_file)

f = open(dt_info_file, "a")

f.write("Dataset Info\n\n")
f.write("Name: " + dt_name + "\n")
f.write("Location: " + dt_location + "\n")
f.write("Number of Images: " + str(dt_total_num_imgs) + "\n")
f.write("Used: " + str(dt_used_imgs) + "\n")
f.write("Training: " + str(dt_train_imgs) + "\n")
f.write("Test: " + str(dt_test_imgs) + "\n")
f.write("Classes: " + str(dt_classes) + "\n")
f.write("\n")
f.write("#################################################\n\n")

###########################################################

instanceSegmentation = True

if instanceSegmentation == True:
    # Instance Segmentation (IS)
    config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    output_dir = "./model/output/instance_segmentation"
    cfg_save_path = "./model/IS_cfg.pickle"
else:
    # Object Detection (OD)
    config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    output_dir = "./model/output/object_detection"
    cfg_save_path = "./model/OD_cfg.pickle"


num_classes = len(dt_classes)
device = "cuda" # "cpu"

train_dataset_name = "LP_train"
train_images_path = os.getcwd() + "/dataset/train"
train_json_annot_path = os.getcwd() + "/dataset/train.json"

test_dataset_name = "LP_test"
test_images_path = os.getcwd() + "/dataset/test"
test_json_annot_path = os.getcwd() + "/dataset/test.json"

save_metadata_dir = os.getcwd() + "/model/dataset_metadata.json"


f.write("Training Info\n\n")
f.write("instanceSegmentation = " + str(instanceSegmentation) + "\n")
f.write("config_file_path = " + config_file_path + "\n")
f.write("checkpoint_url = " + checkpoint_url + "\n")
f.write("output_dir = " + output_dir + "\n")
f.write("cfg_save_path = " + cfg_save_path + "\n\n")

f.write("num_classes = " + str(num_classes) + "\n")
f.write("device = " + device + "\n\n")

f.write("train_dataset_name = " + train_dataset_name + "\n")
f.write("train_images_path = " + train_images_path + "\n")
f.write("train_json_annot_path = " + train_json_annot_path + "\n\n")

f.write("test_dataset_name = " + test_dataset_name + "\n")
f.write("test_images_path = " + test_images_path + "\n")
f.write("test_json_annot_path = " + test_json_annot_path + "\n\n")

f.write("save_metadata_dir = " + save_metadata_dir + "\n\n")


# Register dataset
register_coco_instances(name = train_dataset_name, metadata = {}, json_file = train_json_annot_path, image_root = train_images_path)
register_coco_instances(name = test_dataset_name, metadata = {}, json_file = test_json_annot_path, image_root = test_images_path)

# Verify dataset
plot_samples(dataset_name = train_dataset_name, n = 2)

# Save metadata
save_dataset_metadata(dataset_name = train_dataset_name, save_metadata_dir = save_metadata_dir)


# Get model configuration according to our specifications
cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device, output_dir, f)
# Close file
f.close()

# Save configs with pickle
with open(cfg_save_path, "wb") as f:
    pickle.dump(cfg, f, protocol = pickle.HIGHEST_PROTOCOL)
# Create dir to save model
os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
# Train Model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()





