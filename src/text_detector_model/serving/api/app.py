import flask
import mlflow
import torch
import os
from PIL import Image
import base64
import io
import logging
import numpy as np
import pandas as pd
from torchvision.transforms import v2 as T
from paddleocr import PaddleOCR
from torchvision import tv_tensors

app = flask.Flask(__name__)

# Configure logger
app.logger.setLevel(logging.INFO)
handler = logging.FileHandler("app.log")
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    )
)
app.logger.addHandler(handler)

# Load Faster R-CNN model from MLflow
app.logger.info("Loading Faster R-CNN model from MLflow")
mlflow_run_id = os.environ.get("MLFLOW_RUN_ID")
mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
model_uri = f"runs:/{mlflow_run_id}/{mlflow_model_name}"
detection_model = mlflow.pytorch.load_model(model_uri)
detection_model.eval()

# load PaddleOCR model
app.logger.info("Loading PaddleOCR model")
# TODO try to download the model during docker build and reference it here
recognition_model = PaddleOCR(
    use_angle_cls=True, lang="en", det=False, rec=True, use_gpu=False
)


@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from request
    img_data = base64.b64decode(flask.request.json["image"])
    original_image = Image.open(io.BytesIO(img_data))
    img = tv_tensors.Image(original_image)
    new_height, new_width = 800, 600

    transform = T.Compose(
        [
            T.ToDtype(torch.float, scale=True),
            T.Resize((new_height, new_width)),
            T.ToPureTensor(),
        ]
    )

    app.logger.info("Preprocess image")
    img = transform(img)

    app.logger.info("Run detection")
    # Get predictions
    with torch.no_grad():
        predictions = detection_model([img])[0]

    boxes = predictions["boxes"].numpy().tolist()
    scores = predictions["scores"].numpy().tolist()
    classes = predictions["labels"].numpy().tolist()

    df = pd.DataFrame({"box": boxes, "score": scores, "class": classes})
    # take highest score for each class
    df = df.sort_values(by="score", ascending=False).drop_duplicates(subset="class")
    scale_y = original_image.height / new_height
    scale_x = original_image.width / new_width

    dilatation = 5
    df["original_box"] = df["box"].apply(
        lambda x: [
            int(x[0] * scale_x) - dilatation,
            int(x[1] * scale_y) - dilatation,
            int(x[2] * scale_x) + dilatation,
            int(x[3] * scale_y) + dilatation,
        ]
    )
    app.logger.info(f"detected {len(df)} boxes")

    app.logger.info("Run recognition")
    predicted_text = []
    text_score = []
    for box in df["original_box"]:
        box_image = original_image.crop(box)

        predicted_text_box, text_score_box = recognition_model.ocr(
            np.array(box_image), det=False, rec=True
        )[0][0]
        predicted_text.append(predicted_text_box)
        text_score.append(text_score_box)

    df["predicted_text"] = predicted_text
    df["text_score"] = text_score

    predictions_dict = df[
        ["original_box", "class", "score", "predicted_text", "text_score"]
    ].to_dict(orient="records")

    return flask.jsonify({"predictions": predictions_dict})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
