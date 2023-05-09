#!/usr/bin/env python3

# COMMON LIBRARIES
import os
import cv2
import pickle

from utilsDetectronRoboflow import *
from datetime import datetime

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# EVALUATION
from detectron2.engine import DefaultPredictor

# TRAINING
from detectron2.engine import DefaultTrainer

# ROBOFLOW
from roboflow import Roboflow



# Download dataset from Roboflow
rf = Roboflow(api_key="QO1iBTAWSmIJ28ZyKyVr")
project = rf.workspace("socrob").project("door-handles")
dataset = project.version(7).download("coco-segmentation")


# REGISTER DATASET
DATA_SET_NAME = dataset.name.replace(" ", "-")
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"


# ***(ATTENTION) EDIT THIS SECTION***
DATA_SET_LOCATION = "Roboflow"
DATA_SET_CLASSES = ["doorHandle"]
DATA_SET_NUM_CLASSES = len(DATA_SET_CLASSES)

# TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "train", ANNOTATIONS_FILE_NAME)
TRAIN_DATA_SET_NUM = int(len([entry for entry in os.listdir(TRAIN_DATA_SET_IMAGES_DIR_PATH) if os.path.isfile(os.path.join(TRAIN_DATA_SET_IMAGES_DIR_PATH, entry))]) - 1)

register_coco_instances(
    name=TRAIN_DATA_SET_NAME, 
    metadata={}, 
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH, 
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# TEST SET
TEST_DATA_SET_NAME = f"{DATA_SET_NAME}-test"
TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "test")
TEST_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "test", ANNOTATIONS_FILE_NAME)
TEST_DATA_SET_NUM = int(len([entry for entry in os.listdir(TEST_DATA_SET_IMAGES_DIR_PATH) if os.path.isfile(os.path.join(TEST_DATA_SET_IMAGES_DIR_PATH, entry))]) - 1)

register_coco_instances(
    name=TEST_DATA_SET_NAME, 
    metadata={}, 
    json_file=TEST_DATA_SET_ANN_FILE_PATH, 
    image_root=TEST_DATA_SET_IMAGES_DIR_PATH
)

# VALID SET
VALID_DATA_SET_NAME = f"{DATA_SET_NAME}-valid"
VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "valid")
VALID_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "valid", ANNOTATIONS_FILE_NAME)
VALID_DATA_SET_NUM = int(len([entry for entry in os.listdir(VALID_DATA_SET_IMAGES_DIR_PATH) if os.path.isfile(os.path.join(VALID_DATA_SET_IMAGES_DIR_PATH, entry))]) - 1)

register_coco_instances(
    name=VALID_DATA_SET_NAME, 
    metadata={}, 
    json_file=VALID_DATA_SET_ANN_FILE_PATH, 
    image_root=VALID_DATA_SET_IMAGES_DIR_PATH
)

# Dataset Upload Validation
# print([ data_set for data_set in MetadataCatalog.list() if data_set.startswith(DATA_SET_NAME)])


# TRANING CONFIGS
INSTANCE_SEGMENTATION = True

if INSTANCE_SEGMENTATION:
    # Instance Segmentation (IS)
    ARCHITECTURE = "mask_rcnn_R_50_FPN_3x"
    CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
    OUTPUT_DIR = "./model/output/instance_segmentation"
    CFG_SAVE_PATH = "./model/IS_cfg.pickle"
else:
    # Object Detection (OD)
    ARCHITECTURE = "faster_rcnn_R_50_FPN_3x"
    CONFIG_FILE_PATH = f"COCO-Detection/{ARCHITECTURE}.yaml"
    OUTPUT_DIR = "./model/output/object_detection"
    CFG_SAVE_PATH = "./model/OD_cfg.pickle"


DEVICE = "cuda"


# DATASET VERIFICATION (VISUALIZATION)
plot_samples(dataset_name = TRAIN_DATA_SET_NAME, n = 2)

# SAVE METADATA
MODEL_DIR = os.getcwd() + "/model"
if not os.path.exists(MODEL_DIR):
    print("Creating Model Directory!")
    os.makedirs(MODEL_DIR)

METADATA_DIR = MODEL_DIR + "/dataset_metadata.json"
save_dataset_metadata(dataset_name = TRAIN_DATA_SET_NAME, save_metadata_dir = METADATA_DIR)


# WRITE CONFIGS TO FILE
DATA_SET_INFO_FILE = MODEL_DIR + "/trainingInfo.txt"
if os.path.exists(DATA_SET_INFO_FILE):
  os.remove(DATA_SET_INFO_FILE)


TOTAL_NUM_IMGS = VALID_DATA_SET_NUM + TEST_DATA_SET_NUM + TRAIN_DATA_SET_NUM


f = open(DATA_SET_INFO_FILE, "a")

f.write("Dataset Info\n\n")
f.write("Name: " + DATA_SET_NAME + "\n")
f.write("Location: " + DATA_SET_LOCATION + "\n")
f.write("Number of Images: " + str(TOTAL_NUM_IMGS) + "\n")
f.write("Training: " + str(TRAIN_DATA_SET_NUM) + "\n")
f.write("Validation: " + str(VALID_DATA_SET_NUM) + "\n")
f.write("Test: " + str(TEST_DATA_SET_NUM) + "\n")
f.write("Classes: " + str(DATA_SET_CLASSES) + "\n")
f.write("\n")
f.write("#################################################\n\n")

f.write("Training Info\n\n")
f.write("instanceSegmentation = " + str(INSTANCE_SEGMENTATION) + "\n")
f.write("config_file_path = " + CONFIG_FILE_PATH + "\n")
f.write("checkpoint_url = " + CONFIG_FILE_PATH + "\n")
f.write("output_dir = " + OUTPUT_DIR + "\n")
f.write("cfg_save_path = " + CFG_SAVE_PATH + "\n\n")

f.write("num_classes = " + str(DATA_SET_NUM_CLASSES) + "\n")
f.write("device = " + DEVICE + "\n\n")

f.write("train_dataset_name = " + TRAIN_DATA_SET_NAME + "\n")
f.write("train_images_path = " + TRAIN_DATA_SET_IMAGES_DIR_PATH + "\n")
f.write("train_json_annot_path = " + TRAIN_DATA_SET_ANN_FILE_PATH + "\n\n")

f.write("valid_dataset_name = " + VALID_DATA_SET_NAME + "\n")
f.write("valid_images_path = " + VALID_DATA_SET_IMAGES_DIR_PATH + "\n")
f.write("valid_json_annot_path = " + VALID_DATA_SET_ANN_FILE_PATH + "\n\n")

f.write("test_dataset_name = " + TEST_DATA_SET_NAME + "\n")
f.write("test_images_path = " + TEST_DATA_SET_IMAGES_DIR_PATH + "\n")
f.write("test_json_annot_path = " + TEST_DATA_SET_ANN_FILE_PATH + "\n\n")

f.write("save_metadata_dir = " + METADATA_DIR + "\n\n")


# GET MODEL CONFIGURATION
cfg, f = get_train_cfg(CONFIG_FILE_PATH, TRAIN_DATA_SET_NAME, TEST_DATA_SET_NAME, DATA_SET_NUM_CLASSES, DEVICE, OUTPUT_DIR, f)
f.close()

# SAVE CONFIGS IN PICKLE FILE
with open(cfg_save_path, "wb") as f:
    pickle.dump(cfg, f, protocol = pickle.HIGHEST_PROTOCOL)

# CREATE DIR TO SAVE MODEL
os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

# TRAIN MODEL
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
