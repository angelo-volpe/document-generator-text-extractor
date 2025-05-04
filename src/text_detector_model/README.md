# Text Detector Model

This model is based on Faster RCNN

## Model Architecture
Faster RCNN is an object detection model based using an architecture with different parts as shown in the image:
![Faster R-CNN Architecture](../../images/Faster-RCNN_architecture.webp)

- <b>Backbone</b> is usually a pre-trained an image classification model used to learn image feature, in our case we used mobilenet_v3 that is ligthweight compared to other backbones.
- <b>Region Proposal Network</b> this part of the network is responsible of selecting regions with possible content that will go then in the next part of the network
- <b>Fully Connected Layers</b> this last part classify the region and outputs the score for each class and the bounding box coordinates, the output is replaced in our case to reflect the classes that we have for fine tuning


## Input
The model receives as input an image of a document like and the bounding boxes containing relevant text and the class of each one
![Example Input](../../images/detector_input_example.png)

## Output
The model outputs the predicted boxes with the predicted class and a score

![Example Output](../../images/detector_output_example.png)

## Training
For interactive training you can use the [TextDetectorModel](../../notebooks/TextDetectorModel.ipynb) notebook

in the firs part of the notebook you can specify parameters like:
- `documents_dir`, path to the folder containing data, both images and annotatrions
- `num_epochs`, number of epochs to use for training
- `MLFlow tracking url`, url of MLFlow server for model tracking
- `batch_size`, batch size for training
- `device`, to use for training

Otherwise a job can be used calling the following script with parameters:

```bash
cd ./src/text_detector_model/training
export MLFLOW_TRACKING_URI=http://localhost:5000 # change with your mlflow URI
python main.py --document_id <document_id> --input_base_path <input_base_path> --data_version <data_version>
```

The job is also available via Docker
```bash
cd ./src/text_detector_model/training
docker build . -t document-generator-text-detector:latest
docker run --rm \
           --mount type=bind,source=<local_data_path>,target=/app/data \
           --name document-generator-text-detector \
           --env-file .env.dev \
           document-generator-text-detector:latest --document_id <document_id> --input_base_path <input_base_path> --data_version <data_version>
```

input data for training is expected to follow the structure below:

    └── <input_base_path>
        └── document_<document_id>
            └── <data_version>
                ├── train # folder containing train images
                ├── test  # folder conatining test images
                ├── train_labels.json
                └── test_labels.json



## Serving (API)
Serving via API is ideal if you need to integrate ML model capabilities in an external application that requires prediction on single images with low latency

```bash
cd ./src/text_detector_model/serving/api
export export MLFLOW_TRACKING_URI=http://localhost:5000 # change with your mlflow URI
export MLFLOW_RUN_ID=<run_id>
export MLFLOW_MODEL_NAME=<model_name>
python app.py
```

Using Docker

```bash
cd ./src/text_detector_model/serving/api
docker compose up
```

## Serving (Batch) (TODO)
This is the ideal method in case you need to run predictions on a batch of images