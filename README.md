# 🚀 CNN Binary Image Classification using MobileNet & Streamlit

## 📌 Project Overview
This project is an **end-to-end Computer Vision application** that performs **binary image classification** using a **Convolutional Neural Network (CNN)** with **MobileNet (Transfer Learning)** and is deployed using **Streamlit**.

The goal of this project is to demonstrate how a deep learning image classification model can be:
- trained  
- saved  
- deployed as an interactive web application  

---

## 🧠 Problem Statement
Image classification is a core problem in computer vision with applications in:
- Healthcare
- Manufacturing
- Security
- Automation

### Manual image inspection is:
- ❌ Time-consuming  
- ❌ Error-prone  
- ❌ Not scalable  

This project builds an **automated binary image classifier** that predicts the class of an uploaded image in real time.

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- MobileNet (Transfer Learning)
- NumPy
- Pillow (PIL)
- Streamlit
- Git & GitHub

---

## 🏗️ Project Structure

CNN-Binary-image-classification/
│
├── data/
│ └── raw/
│
├── models/
│ └── binary_mobilenet_model.h5
│
├── src/
│ └── model_mobilenet.py
│
├── test_images/
│
├── app.py
├── model_mobilenet.py
├── requirements.txt
├── README.md
├── .gitignore
└── Dockerfile


---

## 🔍 Model Details
- **Architecture**: MobileNet (Transfer Learning)
- **Input Size**: 224 × 224
- **Output**: Binary classification
- **Loss Function**: Binary Crossentropy
- **Optimizer**: Adam

---

## 🎯 Features
- Upload image via web UI
- Automatic preprocessing
- Real-time prediction
- Displays predicted class with confidence
- Lightweight and fast inference

---

## 🌐 Streamlit Web App
The Streamlit app allows users to:
1. Upload an image
2. Preview the uploaded image
3. Get instant prediction using the trained CNN model

---

## ▶️ Run Locally

### 1️⃣ Clone repository
```bash
git clone https://github.com/poorna-manikanta/CNN-Binary-image-classification.git
cd CNN-Binary-image-classification

## Demo Screenshots
![App Demo] (Screenshots/demo.png)

