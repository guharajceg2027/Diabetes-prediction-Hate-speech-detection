# ML Prediction Dashboard

A comprehensive machine learning web application featuring hate speech detection and diabetes prediction with interactive visualizations.

## Features

- **Hate Speech Detection**: Analyze text content to identify potentially harmful language
- **Diabetes Prediction**: Assess diabetes risk based on medical indicators
- **Interactive Visualizations**: Explore data patterns and model insights
- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Predictions**: Get instant results with confidence scores

## Technology Stack

- **Backend**: Flask, scikit-learn, NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Machine Learning**: Logistic Regression, Random Forest
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Text Processing**: NLTK, TF-IDF Vectorization

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd ml-prediction-app
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
cd backend
python main.py
```

5. **Open your browser**
Navigate to `http://localhost:5000`

## Project Structure

```
ml-prediction-app/
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Flask app initialization
│   │   ├── routes.py            # API routes and endpoints
│   │   └── models.py            # ML models (HateSpeechDetector, DiabetesPredictor)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_preprocessing.py # Data cleaning and preprocessing
│   │   └── visualization.py     # Chart generation utilities
│   └── main.py                  # Application entry point
├── templates/
│   └── index.html              # Main HTML template
├── static/
│   ├── css/
│   │   └── style.css           # Styling and responsive design
│   └── js/
│       └── script.js           # Frontend JavaScript logic
├── models/                     # Saved model files (auto-generated)
├── data/                       # Dataset storage
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Usage Guide

### Hate Speech Detection
1. Navigate to the "Hate Speech Detection" tab
2. Enter text in the textarea or use sample buttons
3. Click "Analyze Text" to get predictions
4. View results with confidence scores

### Diabetes Prediction
1. Go to the "Diabetes Prediction" tab
2. Fill in medical parameters or use sample data
3. Click "Predict Risk" for assessment
4. Review risk level and probability scores

### Visualizations
1. Click on the "Visualizations" tab
2. Press "Load Visualizations" to generate charts
3. Explore data patterns and correlations

## API Endpoints

### POST /api/predict/hate-speech
Analyze text for hate speech content.

**Request Body:**
```json
{
    "text": "Your text to analyze"
}
```

**Response:**
```json
{
    "prediction": "Hate Speech" | "Normal Speech",
    "confidence": 0.85,
    "text": "analyzed text"
}
```

### POST /api/predict/diabetes
Predict diabetes risk based on medical parameters.

**Request Body:**
```json
{
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 80,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 25.5,
    "diabetes_pedigree": 0.5,
    "age": 30
}
```

**Response:**
```json
{
    "prediction": 0 | 1,
    "probability": 0.25,
    "risk_level": "Low" | "Medium" | "High"
}
```

### GET /api/visualizations
Get base64-encoded visualization charts.

## Model Information

### Hate Speech Detection Model
- **Algorithm**: Logistic Regression with TF-IDF Vectorization
- **Features**: Text preprocessing, stopword removal, n-gram analysis
- **Performance**: ~85% accuracy on training data
- **Input**: Raw text string
- **Output**: Binary classification with confidence score

### Diabetes Prediction Model
- **Algorithm**: Random Forest Classifier
- **Features**: 8 medical indicators
- **Performance**: ~80% accuracy on synthetic data
- **Input**: Numerical health parameters
- **Output**: Risk probability and classification

## Development

### Adding New Models
1. Create model class in `backend/app/models.py`
2. Add API endpoint in `backend/app/routes.py`
3. Update frontend in `templates/index.html` and `static/js/script.js`

### Customizing Visualizations
1. Modify `backend/utils/visualization.py`
2. Add new chart types and data analysis
3. Update frontend display logic

### Styling Changes
1. Edit `static/css/style.css`
2. Modify responsive breakpoints
3. Update color schemes and animations

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Change port in main.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

**NLTK data not found:**
```bash
python -c "import nltk; nltk.download('all')"
```

**Module not found errors:**
```bash
pip install --upgrade -r requirements.txt
```

**Visualization not loading:**
- Check browser console for JavaScript errors
- Ensure matplotlib backend is properly configured
- Verify API endpoints are responding

### Performance Optimization
- Use model caching for faster predictions
- Implement request rate limiting
- Add database for storing results
- Use CDN for static assets

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- scikit-learn for machine learning algorithms
- Flask for web framework
- NLTK for natural language processing
- Chart.js and Matplotlib for visualizations
- Font Awesome for icons

## Future Enhancements

- [ ] User authentication and session management
- [ ] Model retraining with new data
- [ ] Export predictions to PDF/CSV
- [ ] Real-time model performance monitoring
- [ ] Integration with external APIs
- [ ] Multi-language support
- [ ] Advanced text preprocessing
- [ ] Ensemble model predictions
- [ ] A/B testing framework
- [ ] Docker containerization

## Contact

For questions or support, please open an issue on GitHub or contact the development team.
```

## Step 9: Create Additional Configuration Files

```txt:requirements.txt
flask==2.3.3
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
nltk==3.8.1
wordcloud==1.9.2
joblib==1.3.2
flask-cors==4.0.0
gunicorn==21.2.0
Pillow==10.0.0
scipy==1.11.1
```

```python:.gitignore
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
models/*.pkl
models/*.joblib
data/*.csv
data/*.json
*.log
```

## Step 10: Running Commands for Windows

Now, let's run the complete setup:

```bash
mkdir ml-prediction-app
```

```bash
cd ml-prediction-app
```

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

Create all the files as shown above, then:

```bash
pip install -r requirements.txt
```

```bash
cd backend
```

```bash
python main.py
```

## Step 11: Testing the Application

Open your browser and go to `http://localhost:5000`

### Test Hate Speech Detection:
1. Click "Sample Hate Speech" button
2. Click "Analyze Text"
3. Observe the results with confidence scores

### Test Diabetes Prediction:
1. Click "High Risk Sample" button
2. Click "Predict Risk"
3. Review the risk assessment

### Test Visualizations:
1. Go to Visualizations tab
2. Click "Load Visualizations"
3. View the generated charts

## Step 12: VS Code Setup

To work efficiently in VS Code:

```bash
code .
```

Install these VS Code extensions:
- Python
- Flask Snippets
- HTML CSS Support
- JavaScript (ES6) code snippets
- Prettier - Code formatter

Create VS Code settings:

```json:.vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/Scripts/python.exe",
    "python.terminal.activateEnvironment": true,
    "files.associations": {
        "*.html": "html"
    },
    "emmet.includeLanguages": {
        "javascript": "javascriptreact"
    },
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black"
}
```

```json:.vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Flask",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/backend/main.py",
            "env": {
                "FLASK_APP": "main.py",
                "FLASK_ENV": "development"
            },
            "args": [],
            "jinja": true,
            "justMyCode": true
        }
    ]
}
-

### ✅ `py -3.11 -m venv venv`

This command **creates a virtual environment** using **Python 3.11**:

* `py -3.11`: Uses Python version 3.11 installed on your system.
* `-m venv venv`: Tells Python to run the `venv` module and create a virtual environment folder named `venv`.

🧠 **Purpose**: Isolate project dependencies (e.g., Flask, scikit-learn) so they don't affect other projects or system Python.

---

### ✅ `.\venv\Scripts\Activate.ps1`

This **activates the virtual environment** on PowerShell (Windows):

* `.\venv\Scripts\Activate.ps1`: Runs the PowerShell script to activate the `venv`.

Once activated, you'll see your prompt change to:

```
(venv) PS C:\Your\Path>
```

🧠 **Purpose**: From now on, when you type `python` or `pip`, it uses the **virtual environment's** Python and packages.

---

### ✅ `python -m backend.main`

This **runs your backend app** (like a Flask server):

* `-m backend.main`: Tells Python to run the `main.py` file inside the `backend` folder as a module.
* Requires that you be in the **parent directory of `backend`** (e.g., `ml-prediction-app`).

🧠 **Purpose**: Starts your backend application (e.g., Flask server).

---

### 🔁 Typical Usage Flow

```bash
# Step 1: Create virtual environment
py -3.11 -m venv venv

# Step 2: Activate it
.\venv\Scripts\Activate.ps1

# Step 3: Install Flask and other dependencies
pip install flask flask-cors scikit-learn pandas numpy

# Step 4: Run your backend app
python -m backend.main
