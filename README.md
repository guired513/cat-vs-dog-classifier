# 🐱🐶 Cat vs Dog Image Classifier

A Convolutional Neural Network (CNN)-based image classification project built with TensorFlow and Flask to predict whether an image is of a cat or a dog. The app features a simple UI for uploading an image, and it returns the predicted class.

## 🚀 Features

- Custom-trained CNN model using TensorFlow/Keras.
- Flask web application with image upload functionality.
- Google Drive integration via `gdown` for downloading model files.
- Designed for deployment on Render.

## 🛠️ Tech Stack

- Python 3.11
- TensorFlow 2.15
- Flask 3.1
- Gunicorn
- Google Drive API (`gdown`)
- Bootstrap (for UI)

## 🧠 Model

- Input size: 150x150 RGB images
- Model type: Sequential CNN
- Output: Binary classification (`Cat`, `Dog`)
- Trained on Kaggle's Cat vs Dog dataset.

## 📁 Project Structure

```
cat-vs-dog-classifier/
├── app.py                  # Flask app
├── templates/              # HTML templates
├── static/                 # Uploaded image preview
├── models/
│   └── cat_dog_classifier.h5  # Pretrained model (downloaded via gdown)
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/guired513/cat-vs-dog-classifier.git
   cd cat-vs-dog-classifier
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   gunicorn app:app
   ```

## 📥 Model Download via GDrive

The model is automatically downloaded from Google Drive at runtime using `gdown`.

## 🌐 Live Deployment

Deployed on [Render](https://render.com)  
(Note: Render’s free tier has a 512Mi memory limit — reduce model size or upgrade plan if needed.)

---

## 🙌 Credits

Developed by **Guillermo V. Red, Jr., DIT**  
Assistant Professor IV, Bicol University  
2025

---

## 📜 License

This project is licensed under the MIT License.