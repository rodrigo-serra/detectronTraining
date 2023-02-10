# Detectron Training
1) Get a dataset and split it in train and test images. Create two separate directories. One for the training and another for testing;

2) Label images using labelme app or any other labeling app. Do it for the train and test images;

3) Run the labelme2coco.py script to convert the train and test images into COCO format. Run in the same directory where the train and test folders are.

    ```bash
    python3 labelme2coco.py train --output train.json
    python3 labelme2coco.py test --output test.json
    ```

    The script will generate the train.json and the test.json files;

4) Run the trainDetectron.py. Make the configurations are the ones you need. Before training the model, we recommend testing the dataset by running the plot_samples() function;

5) Run the trainDetection.py script to test the model on a few images or video;

6) Load the model in the detectron2 node and make sure it works;