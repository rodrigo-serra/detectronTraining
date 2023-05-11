# Detectron Training
1) Label images using the Roboflow Platform (train, test, and validation images);

2) Use the Roboflow API properties to automatically download the dataset to your current directory. Press "Export Dataset" and click on "Continue" to "show download code".

<!-- <div style="display:flex;align-items:center;justify-content:center;margin-bottom:30px">
    <img src="./imgs/roboflow_export_dataset.png" alt="Alt text" title="Optional title" width="50%" style="margin-right:20px">
    <img src="./imgs/roboflow_api.png" alt="Alt text" title="Optional title" width="20%">
</div> -->

Export Dataset in Roboflow            |  Roboflow API
:-------------------------:|:-------------------------:
![](./imgs/roboflow_export_dataset.png)  |  ![](./imgs/roboflow_api.png)

3) Copy the code from the previous step to the script train.py (**DOWNLOAD DATASET FROM ROBOFLOW** section in train.py).

4) Set the model configurations in train.py (**HYPERPARAMETERS** section in train.py).

5) Run the train.py.

6) When the training proccess is done, run the test.py script. This give you an idea of the model performance.

6) Load the model in the detectron2 node and make sure it works;

</br>
Notes: 

- After the training process, your detectronTraining directory should look more or less like this. One folder for the dataset (e.g. Door-Handles-7), and for the model. The latter is the one you should upload to the detectron2_ros package;

- The file **trained_model_info.txt** has all the information regarding training;

```bash
.
├── colab
│   └── colabScript.py
├── Door-Handles-7
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── test
│   │   ├── _annotations.coco.json
│   │   ├── image_129_jpg.rf.7860cfd7ffaa9943a02cfa4c40b0ab2f.jpg
│   │   ├── image_130_jpg.rf.6ae113d91f522025ec7a31a01a325485.jpg
│   │   ├── ...
│   ├── train
│   │   ├── _annotations.coco.json
│   │   ├── Image_00198_jpg.rf.ba6335ad493d8a47f0b0ce52fcba434c.jpg
│   │   ├── image_0_jpg.rf.bd5bafa7ecd4e24cf31393195989b441.jpg
│   │   ├── ...
│   └── valid
│       ├── _annotations.coco.json
│       ├── image_148_jpg.rf.d743315c5a9e81a9a0c9b690becaccc7.jpg
│       ├── image_154_jpg.rf.bbf75497b1057f0732b271197c03acff.jpg
│       ├── ...
├── imgs
│   ├── roboflow_api.png
│   └── roboflow_export_dataset.png
├── labelme_scripts_training
│   ├── labelme2coco.py
│   ├── README.md
│   ├── testDetectron.py
│   ├── trainDetectron.py
│   └── utilsDetectron.py
├── model
│   ├── dataset_metadata.json
│   ├── IS_cfg.pickle
│   ├── trained_model_info.txt
│   └── output
│       └── instance_segmentation
│           ├── events.out.tfevents.1683803640.dolores2.1669923.0
│           ├── last_checkpoint
│           ├── metrics.json
│           └── model_final.pth
├── README.md
├── test.py
└── train.py
```