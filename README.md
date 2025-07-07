# 😷 Face Mask Detector

This is my Final Project for the **Digital Image Processing (DIP)** course (6th Semester).  
It is a real-time **Face Mask Detection System** built using **Python**, **OpenCV**, and a **CNN model**.

The application captures live video from a webcam and detects whether the person is wearing a mask or not. It displays a **bounding box** around the face with a label:

- 🟩 **With Mask** (Green Box)
- 🟥 **Without Mask** (Red Box)

---

## 📸 Features

- Real-time face mask detection using webcam
- Bounding boxes and label predictions on live faces
- Uses a custom-trained CNN model for classification
- Fast and accurate predictions

---

## 🛠️ Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- Haar Cascade (for face detection)
- Pre-trained CNN model (`mask_detection_model.h5`)
- Visual Studio Code

---

## 📁 Dataset

The dataset used for training was downloaded from **Kaggle**:  
🔗 [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection)

> ⚠️ Due to file size limitations, the dataset is **not included** in this repository.

To retrain the model, download the dataset and place it in a folder named `dataset/`.

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/Motasim-Naranja/Face-Mask-Detector.git
cd Face-Mask-Detector
Activate the virtual environment:

bash
Copy
Edit
venv\Scripts\activate  # For Windows
Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Run the live mask detection app:

bash
Copy
Edit
python mask_detection_realtime.py
✅ Ensure your webcam is connected.

📚 Learning Outcomes
Implemented image classification using CNN

Learned practical application of face detection and real-time video processing

Gained hands-on experience with OpenCV and model deployment

Developed a structured and functional project from scratch

📄 Project Report
A detailed project report (Mask_Detection_Project_Report.docx) is included for academic documentation.

👨‍💻 Author
Motasim Ejaz
6th Semester BSCS Student
Sindh Madressatul Islam University (SMIU)
📍 Karachi, Pakistan

⭐ If you found this helpful or inspiring, feel free to star the repo and share it with others!
