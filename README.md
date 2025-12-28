# 🪙 💵 Fake Currency Detection System using Deep Learning

A full-stack AI-based Fake Currency Detection Web Application that detects whether an uploaded Indian currency note is Real or Fake, along with its Denomination, using Deep Learning (CNN) and Flask.



🚀 Project Highlights


✅ Detects whether uploaded image is a currency note or not

✅ Identifies Indian currency denomination (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)

✅ Classifies note as Real or Fake

✅ Displays confidence score

✅ Simple & clean Flask-based web UI

✅ Trained on custom dataset (7000+ images)



👨‍💻 **Developed by:** Meet Shah

---

## 🚀 Features
- Detects whether an image is a **currency note or not**
- Predicts **denomination** (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)
- Classifies note as **Real or Fake**
- Confidence score shown
- Flask-based web interface
- Uses trained CNN models (`.keras`)

---

## 🧠 Tech Stack
- Python 3.10
- TensorFlow / Keras
- OpenCV
- NumPy
- Flask
- HTML, CSS

---
Each denomination contains real and fake currency note images collected from multiple sources and angles.
## 🏗️ Model Architecture

### 1️⃣ Denomination Classification Model
- CNN-based multi-class classifier
- Input size: 224 × 224 × 3
- Output classes: 6 (₹10 – ₹500)

### 2️⃣ Fake Currency Detection Model
- Binary CNN classifier
- Output: REAL / FAKE with confidence



## 📁 Project Structure

project structure and dataset structure screenshot also provided in screenshot folder check it for better understanding

fake-currency-detector/
│
├── app/
│ ├── app.py
│ ├── static/
│ │        └──style.css
│ │        └── uploads/
│ └── templates/
│           └──index.html
│
├── models/
│ ├── denomination_model.keras
│ └── fake_500.keras
│
├── training/
│ ├── train_denomination.py
│ └── train_fake_500.py
│
├── venv (not uploaded system specific provided requirement.txt)
├── data/ (not uploaded)
│
├── requirements.txt
├── README.md
└── .gitignore

# 📁 Dataset Structure
data/
├── 10/
│ ├── real/
│ └── fake/
├── 20/
├── 50/
├── 100/
├── 200/
└── 500/


---

## 📊 Dataset
- Dataset sourced from **Kaggle**
- link of dataset : https://www.kaggle.com/datasets/preetrank/indian-currency-real-vs-fake-notes-dataset?resource=download&select=data
- Contains real and fake Indian currency notes
- Dataset is **not uploaded** to GitHub due to size constraints

## 🔹 Pre-trained Models

Due to GitHub file size limits, trained models are not included.

Download models from:
- denomination_model.keras → [download link : https://drive.google.com/drive/folders/1lu_5ySHSlbASnGZoX6TkJ9kIrrh6EpL7?usp=sharing]
- fake_500.keras → [download link : https://drive.google.com/drive/folders/1lu_5ySHSlbASnGZoX6TkJ9kIrrh6EpL7?usp=sharing]

Place both files inside:
models/



---
## 🚀 How to Run the Project

### Step 1: Clone Repository
```bash
git clone https://github.com/meetshah7572/fake-currency-detector.git
cd fake-currency-detector

### Step 2: create virtual environment
python -m venv venv
venv\Scripts\activate

step 3 : install requirements
pip install -r requirements.txt


step 4 : run application 
python app/app.py

step 5 : open browser 
http://127.0.0.1:5000/

Upload a currency note image and get instant results.

⚠️ Limitations

Performance depends on image quality, angle, and lighting

Extremely blurred or cropped images may give uncertain results

Designed for academic and demonstration purposes
## 📸 Application Screenshots

### 🏠 Home Page
Upload interface for currency detection.

![Home Page](screenshots/home.png)

---

### ✅ Real Currency Detection
Correctly identifies real currency note with denomination and confidence.

![Real Note](screenshots/real_note.png)

---

### ❌ Fake Currency Detection
Detects fake note with confidence score.

![Fake Note](screenshots/fake_note.png)

---

### 🚫 Non-Currency Image Detection
Rejects random images that are not currency notes.

![Not Currency](screenshots/not_currency.png)



👨‍💻 Author

Meet Shah
M.Tech – Artificial Intelligence
Pandit Deendayal Energy University (PDEU)

📜 License

This project is for educational and research purposes only.