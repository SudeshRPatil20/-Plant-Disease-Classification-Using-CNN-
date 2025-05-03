# 🌿 Plant Disease Classification Using CNN

This project uses deep learning to classify plant diseases from leaf images. It is built with a Convolutional Neural Network (CNN) for model training and Streamlit for interactive web-based predictions.

---

## 📌 Overview

Plants are often exposed to various diseases that can affect yield and quality. Early detection is crucial for preventing spread and ensuring crop health. This project:

- Uses a CNN model to identify plant diseases.
- Trains on a large dataset of labeled leaf images.
- Deploys a web app using Streamlit where users can upload images and get instant predictions.

---

## 📁 Dataset

We use the **"New Plant Diseases Dataset"** available on Kaggle.

🔗 [Download from Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

### Instructions to Download

1. Go to the dataset page linked above.
2. Log into your Kaggle account.
3. Click **Download** to get the zip file.
4. Extract the contents and place the dataset folder (e.g., `New Plant Diseases Dataset(Augmented)/`) into the project directory under `./dataset/`.

---

## 🧠 Model Training

Model training is done using a CNN architecture inside a Jupyter Notebook.

### Steps:

1. Open the notebook:
   ```bash
   jupyter notebook plant_disease_training.ipynb


##### 🧠 Features
Upload a leaf image.

The model classifies it into one of the plant disease categories.

Displays prediction label and confidence score.

##### 🗂️ Project Structure
bash
Copy
Edit
plant_desies_detection/
│
├── app.py                         # Streamlit web app
├── plant_disease_training.ipynb  # Jupyter notebook for training the model
├── model.h5                       # Trained CNN model file
├── dataset/                       # Folder containing Kaggle dataset
│   └── New Plant Diseases Dataset(Augmented)/
├── requirements.txt               # List of dependencies
└── README.md                      # Project documentation
##### 🛠️ Installation Guide
Clone the Repository
bash
Copy
Edit
git clone https://github.com/SudeshRPatil20/-Plant-Disease-Classification-Using-CNN-.git
cd plant_desies_detection
Install Python Dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ Ensure you have TensorFlow-compatible hardware (preferably with GPU) for efficient training.

##### 📋 Requirements
Python 3.7+

TensorFlow / Keras

NumPy

Matplotlib

Pandas

OpenCV

scikit-learn

Streamlit

##### 💡 If you don’t have a requirements.txt yet, create one with:

bash
Copy
Edit
pip freeze > requirements.txt
🧪 Model Training Steps
In plant_disease_training.ipynb, follow these steps:

Load and preprocess the data

Train the CNN model

Evaluate performance

Save the trained model as model.h5

##### 🧪 Example Usage
After launching the app, you'll see a UI to upload an image. Upload a leaf image, and the model will display the predicted disease name along with its confidence.

#### ✅ Results
The CNN model achieves strong performance on both training and validation datasets. It generalizes well to unseen images, although performance depends on:

Quality and quantity of data

Preprocessing and augmentation techniques

##### 🔮 Future Improvements
Integrate real-time camera input

Expand dataset to more plant species

Deploy on mobile using TensorFlow Lite

Enhance UI/UX with more features

