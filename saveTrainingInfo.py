#!/usr/bin/env python3

import os

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
                  cfg_path):

    FILE_NAME = model_dir + '/' + filename

    dt_classes = dictionary["thing_classes"]

    dt_train_imgs = int(len([entry for entry in os.listdir(train_data_set_images_dir_path) if os.path.isfile(os.path.join(train_data_set_images_dir_path, entry))]) - 1)
    dt_test_imgs = int(len([entry for entry in os.listdir(test_data_set_images_dir_path) if os.path.isfile(os.path.join(test_data_set_images_dir_path, entry))]) - 1)
    dt_valid_imgs = int(len([entry for entry in os.listdir(valid_data_set_images_dir_path) if os.path.isfile(os.path.join(valid_data_set_images_dir_path, entry))]) - 1)

    dt_total_num_imgs = dt_train_imgs + dt_test_imgs + dt_valid_imgs

    f = open(FILE_NAME, "a")

    f.write("Dataset Info\n\n")
    f.write("Name: " + data_set_name + "\n")
    f.write("Location: Roboflow\n")
    f.write("Number of Images: " + str(dt_total_num_imgs) + " images\n")
    f.write("training: " + str(dt_train_imgs) + " images\n")
    f.write("test: " + str(dt_test_imgs) + " images\n")
    f.write("valid: " + str(dt_valid_imgs) + " images\n")
    f.write("Classes: " + str(dt_classes) + "\n")
    f.write("\n")
    f.write("#################################################\n\n")

    f.write("training Info\n\n")
    f.write("config_file_path = " + config_file_path + "\n")
    f.write("checkpoint_url = " + config_file_path + "\n")
    f.write("output_dir = " + output_dir_path + "\n")
    f.write("cfg_save_path = " + cfg_path + "\n\n")

    f.write("num_classes = " + str(len(dictionary["thing_classes"])) + "\n")

    f.write("train_dataset_name = " + train_data_set_name + "\n")
    f.write("train_images_path = " + train_data_set_images_dir_path + "\n")
    f.write("train_json_annot_path = " + train_data_set_ann_file_path + "\n\n")

    f.write("valid_dataset_name = " + valid_data_set_name + "\n")
    f.write("valid_images_path = " + valid_data_set_images_dir_path + "\n")
    f.write("valid_json_annot_path = " + valid_data_set_ann_file_path + "\n\n")

    f.write("test_dataset_name = " + test_data_set_name + "\n")
    f.write("test_images_path = " + test_data_set_images_dir_path + "\n")
    f.write("test_json_annot_path = " + test_data_set_ann_file_path + "\n\n")

    f.write("save_metadata_dir = " + metadata_dir + "\n\n")

    f.write("cfg.DATALOADER.NUM_WORKERS = " + str(cfg.DATALOADER.NUM_WORKERS) + "\n")
    f.write("cfg.SOLVER.IMS_PER_BATCH = " + str(cfg.SOLVER.IMS_PER_BATCH) + "\n")
    f.write("cfg.SOLVER.BASE_LR = " + str(cfg.SOLVER.BASE_LR) + "\n")
    f.write("cfg.SOLVER.MAX_ITER = " + str(cfg.SOLVER.MAX_ITER) + "\n")
    f.write("cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = " + str(cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE) + "\n\n")
    f.write("cfg.TEST.EVAL_PERIOD = " + str(cfg.TEST.EVAL_PERIOD) + "\n")
    f.write("cfg.INPUT.MASK_FORMAT = " + cfg.INPUT.MASK_FORMAT + "\n")







