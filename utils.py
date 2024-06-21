#!/usr/bin/env python3

import os
from roboflow import Roboflow

def downloadDataset():
    # DOWNLOAD DATASET FROM ROBOFLOW
    rf = Roboflow(api_key="QO1iBTAWSmIJ28ZyKyVr")
    project = rf.workspace("socrob").project("testbed_isr")
    version = project.version(2)
    dataset = version.download("coco-segmentation")
    return dataset


def savetrainInfo(filename, 
                  dictionary, 
                  cfg, 
                  data_set_name, 
                  model_dir, 
                  train_data_set_name, 
                  train_data_set_images_dir_path, 
                  train_data_set_ann_file_path, 
                  test_data_set_name, 
                  test_data_set_images_dir_path, 
                  test_data_set_ann_file_path, 
                  valid_data_set_name, 
                  valid_data_set_images_dir_path,
                  valid_data_set_ann_file_path,
                  metadata_dir,
                  config_file_path, 
                  output_dir_path, 
                  cfg_path,
                  train_config):

    FILE_NAME = model_dir + '/' + filename

    dt_classes = dictionary["thing_classes"]

    if os.path.exists(train_data_set_images_dir_path):
        dt_train_imgs = int(len([entry for entry in os.listdir(train_data_set_images_dir_path) if os.path.isfile(os.path.join(train_data_set_images_dir_path, entry))]) - 1)
    else:
        dt_train_imgs = 0

    if os.path.exists(valid_data_set_images_dir_path):
        dt_valid_imgs = int(len([entry for entry in os.listdir(valid_data_set_images_dir_path) if os.path.isfile(os.path.join(valid_data_set_images_dir_path, entry))]) - 1)
    else:
        dt_valid_imgs = 0

    if os.path.exists(test_data_set_images_dir_path):
        dt_test_imgs = int(len([entry for entry in os.listdir(test_data_set_images_dir_path) if os.path.isfile(os.path.join(test_data_set_images_dir_path, entry))]) - 1)
    else:
        dt_test_imgs = 0

    dt_total_num_imgs = dt_train_imgs + dt_test_imgs + dt_valid_imgs

    f = open(FILE_NAME, "a")

    f.write("## DATASET INFO\n")
    f.write("Name: " + data_set_name + "\n")
    f.write("Location: Roboflow\n")
    f.write("Number of Images: " + str(dt_total_num_imgs) + " images\n")
    f.write("training: " + str(dt_train_imgs) + " images\n")
    f.write("test: " + str(dt_test_imgs) + " images\n")
    f.write("valid: " + str(dt_valid_imgs) + " images\n")
    f.write("Classes: " + str(dt_classes) + "\n")
    f.write("num_classes = " + str(len(dictionary["thing_classes"])) + "\n\n")
    
    f.write("## HYPERPARAMETERS\n")
    f.write("ARCHITECTURE = " + train_config["ARCHITECTURE"] + "\n")
    f.write("NUM_WORKERS = " + str(train_config["NUM_WORKERS"]) + "\n")
    f.write("USE_WARMUP = " + str(train_config["USE_WARMUP"]) + "\n")
    f.write("WARMUP_ITERS = " + str(train_config["WARMUP_ITERS"]) + "\n")
    f.write("IMS_PER_BATCH = " + str(train_config["IMS_PER_BATCH"]) + "\n")
    f.write("BASE_LR = " + str(train_config["BASE_LR"]) + "\n")
    f.write("MAX_ITER = " + str(train_config["MAX_ITER"]) + "\n")
    f.write("STEPS = " + "[" + str(train_config["STEPS"][0]) + "," + str(train_config["STEPS"][1]) + "]" + "\n")
    f.write("GAMMA = " + str(train_config["GAMMA"]) + "\n")
    f.write("CHECKPOINT_PERIOD = " + str(train_config["CHECKPOINT_PERIOD"]) + "\n")
    f.write("USE_AMP = " + str(train_config["USE_AMP"]) + "\n")
    f.write("AMP_ENABLED = " + str(train_config["AMP_ENABLED"]) + "\n")
    f.write("BATCH_SIZE_PER_IMAGE = " + str(train_config["BATCH_SIZE_PER_IMAGE"]) + "\n")
    f.write("EVAL_PERIOD = " + str(train_config["EVAL_PERIOD"]) + "\n")
    f.write("MASK_FORMAT = " + str(train_config["MASK_FORMAT"]) + "\n")







