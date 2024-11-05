#!/usr/bin/env python3

# COMMON LIBRARIES
import os
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PARAMETERS & HYPERPARAMETERS CONFIG
f = open("config.json")
train_config = json.load(f)

RESUME_TRAINING = True
if train_config["RESUME_TRAINING"] == "False":
    RESUME_TRAINING = False

SHOW_IMAGE = True
if train_config["SHOW_IMAGE"] == "False":
    SHOW_IMAGE = False

MODEL_DIR = os.getcwd() + "/model"
ARCHITECTURE = train_config["ARCHITECTURE"]
CONFIG_FILE_PATH = f"COCO-InstanceSegmentation/{ARCHITECTURE}.yaml"
NUM_WORKERS = train_config["NUM_WORKERS"]
WARMUP_ITERS = train_config["WARMUP_ITERS"]
WARMUP_FACTOR = 1.0 / 1000
IMS_PER_BATCH = train_config["IMS_PER_BATCH"]
BASE_LR = train_config["BASE_LR"]
MAX_ITER = train_config["MAX_ITER"]
STEPS = (train_config["STEPS"][0], train_config["STEPS"][1])
GAMMA = train_config["GAMMA"]
CHECKPOINT_PERIOD = train_config["CHECKPOINT_PERIOD"]

AMP_ENABLED = True
if train_config["AMP_ENABLED"] == "False":
    AMP_ENABLED = False

BATCH_SIZE_PER_IMAGE = train_config["BATCH_SIZE_PER_IMAGE"]
EVAL_PERIOD = train_config["EVAL_PERIOD"]
MASK_FORMAT = train_config["MASK_FORMAT"]
CFG_PATH = os.getcwd() + "/model/IS_cfg.pickle"
OUTPUT_DIR_PATH = "model/output/instance_segmentation"


# READ TRAINING LOGS
file_path = "model/output/instance_segmentation/metrics.json"
training_res = []
with open(file_path, 'r') as file:
    for line in file:
        try:
            training_res.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Could not decode line: {line}")


# COMPUTE EVAL ITETARIONS BASED ON MAX_ITER AND EVAL_PERIOD
num_of_evaluations = int(MAX_ITER / EVAL_PERIOD)
eval_iter = -1
eval_iter_arr = []
for i in range (0, num_of_evaluations):
    eval_iter += EVAL_PERIOD
    eval_iter_arr.append(eval_iter)
    

# GET PARAMS OF EVAL ITERATIONS
dic_eval_res = []
for element in training_res:
        if element['iteration'] in eval_iter_arr:
            log = element
            dic_eval_res.append(element)


dic_eval_res.append(training_res[len(training_res) - 1])


# WRITE EVAL LOGS TO CSV FILE
headers = ['iteration', 'bbox/AP', 'segm/AP', 'total_loss']
filename = 'model/output/instance_segmentation/metrics_analyse.csv'

with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for element in dic_eval_res:
            writer.writerow({field: element.get(field) for field in headers})



# COMPUTE AVG TOTAL LOSS BETWEEN EVAL PERIODS
total_losses = {period: [] for period in eval_iter_arr}
for record in training_res:
    if 'iteration' in record and 'total_loss' in record:
        iteration = record['iteration']
        total_loss = record['total_loss']
        for period in eval_iter_arr:
            if iteration <= period:
                total_losses[period].append(total_loss)
                break

    avg_total_losses = {period: np.mean(losses) for period, losses in total_losses.items() if losses}



# WRITE TO TRAINING INFO FILE
output_file = os.getcwd() + '/model/trained_model_info.txt'

# Write average losses
with open(output_file, 'a') as file:
    file.write("\n## AVERAGES OF TOTAL LOSS FOR EACH EVALUATION PERIOD:\n")
    for period, avg_loss in avg_total_losses.items():
        file.write(f"Evaluation period: {period}, Average total_loss: {avg_loss}\n")
    file.write("\n")

# Write eval periods logs
with open(filename, mode='r') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    file_contents = list(csv_reader)

    file_contents_str = [
        f"Iteration: {row['iteration']}, bbox/AP: {row['bbox/AP']}, segm/AP: {row['segm/AP']}, total_loss: {row['total_loss']}\n"
        for row in file_contents
    ]

    with open(output_file, 'a') as file:
        file.write("## EVAL PERIOD RESULTS (metrics_analyse.csv):\n")
        file.writelines(file_contents_str)
        file.write("\n")


## PLOT RESULTS TO A GET BETTER IDEA OF THE RESULTS
# Step 1: Create a dir to save plots
plots_dir = os.getcwd() + '/model/plots'
os.makedirs(plots_dir, exist_ok=True)

# Step 2: Convert the data to a DataFrame
df = pd.DataFrame(training_res)

# Step 3: Plot the Data
# Plot total loss over iterations
plt.figure(figsize=(12, 6))
plt.plot(df['iteration'], df['total_loss'], label='Total Loss', color='blue')
plt.xlabel('Iteration')
plt.ylabel('Total Loss')
plt.title('Total Loss over Iterations')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'total_loss_over_iterations.png'))
plt.close()

# Plot average precision over iterations
plt.figure(figsize=(12, 6))
plt.plot(df['iteration'], df['fast_rcnn/cls_accuracy'], label='Fast R-CNN Classification Accuracy', color='green')
plt.plot(df['iteration'], df['mask_rcnn/accuracy'], label='Mask R-CNN Accuracy', color='red')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy over Iterations')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'accuracy_over_iterations.png'))
plt.close()

# Additional Plots: False Negatives and Positives

# Plot False Negatives over iterations
plt.figure(figsize=(12, 6))
plt.plot(df['iteration'], df['mask_rcnn/false_negative'], label='Mask R-CNN False Negative', color='purple')
plt.plot(df['iteration'], df['fast_rcnn/false_negative'], label='Fast R-CNN False Negative', color='orange')
plt.xlabel('Iteration')
plt.ylabel('False Negative')
plt.title('False Negatives over Iterations')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'false_negatives_over_iterations.png'))
plt.close()

# Plot False Positives over iterations
plt.figure(figsize=(12, 6))
plt.plot(df['iteration'], df['mask_rcnn/false_positive'], label='Mask R-CNN False Positive', color='brown')
plt.xlabel('Iteration')
plt.ylabel('False Positive')
plt.title('False Positives over Iterations')
plt.legend()
plt.savefig(os.path.join(plots_dir, 'false_positives_over_iterations.png'))
plt.close()

# Plot Loss Components
components = ['loss_box_reg', 'loss_cls', 'loss_mask', 'loss_rpn_cls', 'loss_rpn_loc']
plt.figure(figsize=(12, 12))
for i, component in enumerate(components, 1):
    plt.subplot(len(components), 1, i)
    plt.plot(df['iteration'], df[component], label=component)
    plt.xlabel('Iteration')
    plt.ylabel(component)
    plt.title(f'{component} over Iterations')
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'loss_components_over_iterations.png'))
plt.close()