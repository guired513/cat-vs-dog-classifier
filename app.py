from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown

app = Flask(__name__)

model_path = "models/cat_dog_classifier.h5"
drive_file_id = "1AMDB2uRqLr3rUsH1jOi3-ZF7Wdi-ozlY"

if not os.path.exists(model_path):
    os.makedirs("models", exist_ok=True)
    url = f"https://drive.google.com/uc?id={drive_file_id}"
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=0)
    prediction = model.predict(img_tensor)[0][0]
    return "Dog" if prediction > 0.5 else "Cat"

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            filename = file.filename
    return render_template("index.html", prediction=prediction, filename=filename)

if __name__ == "__main__":
    app.run()