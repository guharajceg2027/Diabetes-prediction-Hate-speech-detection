
# 🤖 ML Prediction Dashboard

A smart and user-friendly web application that brings machine learning predictions to life. It features real-time hate speech detection and diabetes risk analysis — all wrapped in an interactive, responsive interface.

---

## ✨ Key Highlights

* 🧠 **Hate Speech Detection**: Instantly identify offensive or harmful language in text input.
* 🩺 **Diabetes Prediction**: Predict the likelihood of diabetes based on medical parameters.
* 📊 **Interactive Visualizations**: Discover insights with beautiful, auto-generated charts.
* 📱 **Responsive UI**: Seamlessly works across desktops, tablets, and mobile devices.
* ⚡ **Real-Time Predictions**: Instant ML results with clear confidence scores.

---

## 🛠️ Tech Stack

**Backend**

* Flask
* scikit-learn
* NumPy & Pandas

**Frontend**

* HTML5, CSS3, Vanilla JS (ES6+)

**ML Models**

* Logistic Regression
* Random Forest

**Data Viz**

* Matplotlib, Seaborn, Plotly

**Text Processing**

* NLTK
* TF-IDF

---

## 🚀 Getting Started

### 🔧 Prerequisites

* Python 3.8+
* Git & pip

### 📦 Installation

```bash
git clone https://github.com/guharajceg2027/ml-prediction-app.git
cd ml-prediction-app
python -m venv venv
venv\Scripts\activate       # Windows
# or: source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
cd backend
python main.py
```

Then visit: `http://localhost:5000`

---

## 🗂️ Project Overview

```
ml-prediction-app/
├── backend/
│   ├── app/              # Routes & model logic
│   ├── utils/            # Preprocessing & chart tools
│   └── main.py           # App entry point
├── static/               # CSS and JS
├── templates/            # HTML templates
├── data/                 # Datasets
├── models/               # Trained model files
├── requirements.txt      # Dependencies
└── README.md             # You're here!
```

---

## 👩‍🔬 How to Use

### 🗣️ Hate Speech Detector

* Go to the Hate Speech tab
* Type/paste your text
* Click **Analyze Text**
* View prediction and confidence level

### 🧬 Diabetes Predictor

* Enter medical values (or use sample)
* Click **Predict Risk**
* View diabetes risk and probability

### 📈 Visualizations

* Navigate to **Visualizations**
* Click **Load Visuals**
* Analyze charts and trends

---

## 🔌 API Reference

### 📩 `POST /api/predict/hate-speech`

**Request**

```json
{ "text": "This is a sample input" }
```

**Response**

```json
{ "prediction": "Normal Speech", "confidence": 0.93 }
```

### 📩 `POST /api/predict/diabetes`

**Request**

```json
{ "glucose": 110, "bmi": 25.6, "age": 45, ... }
```

**Response**

```json
{ "prediction": 1, "risk_level": "High", "probability": 0.81 }
```

### 📤 `GET /api/visualizations`

Returns base64-encoded chart images.

---

## 📊 Model Details

### Hate Speech Detection

* **Model**: Logistic Regression + TF-IDF
* **Accuracy**: \~85%
* **Input**: Raw text
* **Output**: Hate/Normal with confidence

### Diabetes Prediction

* **Model**: Random Forest
* **Features**: 8 health metrics
* **Accuracy**: \~80%
* **Output**: Risk prediction + probability

---

## 🧩 Extending the App

### Add a New ML Model

* Define it in `models.py`
* Add route in `routes.py`
* Connect it to frontend JS

### Customize Charts

* Edit `visualization.py` in `utils/`
* Add new charts (bar, line, heatmap...)

### Style Your UI

* Modify `static/css/style.css`
* Adjust layouts, colors, animations

---

## 🧯 Troubleshooting

**Port already in use**

```bash
# In backend/main.py
app.run(port=5001)
```

**NLTK errors?**

```bash
python -c "import nltk; nltk.download('all')"
```

**Model files not found?**

* Make sure `/models` directory exists
* Or retrain and save model manually

---

## 🏃‍♂️ Performance Tips

* Use joblib to cache models
* Enable gzip compression
* Add backend rate-limiting
* Host static files via CDN

---

## 🤝 Contributing

1. Fork the repo
2. Create a new branch `git checkout -b feature/xyz`
3. Make changes and commit
4. Push and open a PR!

---

## 🙏 Acknowledgments

Thanks to these awesome tools:

* [Flask](https://flask.palletsprojects.com/)
* [scikit-learn](https://scikit-learn.org/)
* [NLTK](https://www.nltk.org/)
* [Matplotlib](https://matplotlib.org/)
* [Plotly](https://plotly.com/)

---

## 🔮 What’s Next?

* [ ] Add user login & session
* [ ] Export prediction results
* [ ] Monitor model performance
* [ ] Multi-language support
* [ ] Deploy using Docker

