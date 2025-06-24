// Global variables
let currentTab = 'hate-speech';

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('ML Prediction Dashboard loaded');
    showTab('hate-speech');
});

// Tab navigation functionality
function showTab(tabName) {
    // Hide all tab contents
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(tabName);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Add active class to clicked button
    const clickedButton = event ? event.target : document.querySelector(`[onclick="showTab('${tabName}')"]`);
    if (clickedButton) {
        clickedButton.classList.add('active');
    }
    
    currentTab = tabName;
}

// Hate Speech Detection
async function predictHateSpeech() {
    const textInput = document.getElementById('text-input');
    const resultContainer = document.getElementById('hate-speech-result');
    
    const text = textInput.value.trim();
    
    if (!text) {
        showError(resultContainer, 'Please enter some text to analyze.');
        return;
    }
    
    // Show loading
    showLoading(resultContainer, 'Analyzing text...');
    
    try {
        const response = await fetch('/api/predict/hate-speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayHateSpeechResult(resultContainer, data);
        } else {
            showError(resultContainer, data.error || 'An error occurred during prediction.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(resultContainer, 'Network error. Please try again.');
    }
}

function displayHateSpeechResult(container, data) {
    const isHateSpeech = data.prediction === 'Hate Speech';
    const confidence = (data.confidence * 100).toFixed(1);
    
    const resultClass = isHateSpeech ? 'result-danger' : 'result-success';
    const icon = isHateSpeech ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle';
    const confidenceColor = isHateSpeech ? '#dc3545' : '#28a745';
    
    container.innerHTML = `
        <div class="animate-fade-in">
            <h3><i class="${icon}"></i> Analysis Result</h3>
            <div class="result-item">
                <strong>Prediction:</strong> ${data.prediction}
            </div>
            <div class="confidence-meter">
                <label>Confidence Level:</label>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${confidence}%; background: ${confidenceColor};">
                        ${confidence}%
                    </div>
                </div>
            </div>
            <div class="result-item">
                <strong>Analyzed Text:</strong> "${data.text}"
            </div>
        </div>
    `;
    
    container.className = `result-container ${resultClass} show`;
}

// Diabetes Prediction
async function predictDiabetes() {
    const resultContainer = document.getElementById('diabetes-result');
    
    // Get form values
    const formData = {
        pregnancies: document.getElementById('pregnancies').value,
        glucose: document.getElementById('glucose').value,
        blood_pressure: document.getElementById('blood-pressure').value,
        skin_thickness: document.getElementById('skin-thickness').value,
        insulin: document.getElementById('insulin').value,
        bmi: document.getElementById('bmi').value,
        diabetes_pedigree: document.getElementById('diabetes-pedigree').value,
        age: document.getElementById('age').value
    };
    
    // Validate inputs
    const validation = validateDiabetesInputs(formData);
    if (!validation.isValid) {
        showError(resultContainer, validation.message);
        return;
    }
    
    // Show loading
    showLoading(resultContainer, 'Calculating diabetes risk...');
    
    try {
        const response = await fetch('/api/predict/diabetes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayDiabetesResult(resultContainer, data, formData);
        } else {
            showError(resultContainer, data.error || 'An error occurred during prediction.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError(resultContainer, 'Network error. Please try again.');
    }
}

function validateDiabetesInputs(data) {
    const errors = [];
    
    if (data.glucose < 0 || data.glucose > 300) {
        errors.push('Glucose level should be between 0-300 mg/dL');
    }
    
    if (data.blood_pressure < 0 || data.blood_pressure > 200) {
        errors.push('Blood pressure should be between 0-200 mmHg');
    }
    
    if (data.bmi < 0 || data.bmi > 50) {
        errors.push('BMI should be between 0-50');
    }
    
    if (data.age < 1 || data.age > 120) {
        errors.push('Age should be between 1-120 years');
    }
    
    return {
        isValid: errors.length === 0,
        message: errors.join(', ')
    };
}

function displayDiabetesResult(container, data, inputData) {
    const hasDiabetes = data.prediction === 1;
    const probability = (data.probability * 100).toFixed(1);
    const riskLevel = data.risk_level;
    
    let resultClass, riskClass, icon, riskColor;
    
    switch(riskLevel) {
        case 'High':
            resultClass = 'result-danger';
            riskClass = 'risk-high';
            icon = 'fas fa-exclamation-triangle';
            riskColor = '#dc3545';
            break;
        case 'Medium':
            resultClass = 'result-warning';
            riskClass = 'risk-medium';
            icon = 'fas fa-exclamation-circle';
            riskColor = '#ffc107';
            break;
        default:
            resultClass = 'result-success';
            riskClass = 'risk-low';
            icon = 'fas fa-check-circle';
            riskColor = '#28a745';
    }
    
    container.innerHTML = `
        <div class="animate-fade-in">
            <h3><i class="${icon}"></i> Diabetes Risk Assessment</h3>
            <div class="result-item">
                <strong>Prediction:</strong> ${hasDiabetes ? 'High Risk of Diabetes' : 'Low Risk of Diabetes'}
            </div>
            <div class="result-item">
                <strong>Risk Level:</strong> <span class="risk-indicator ${riskClass}">${riskLevel} Risk</span>
            </div>
            <div class="confidence-meter">
                <label>Probability:</label>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${probability}%; background: ${riskColor};">
                        ${probability}%
                    </div>
                </div>
            </div>
            <div class="feature-importance">
                <h4>Input Summary:</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-top: 10px;">
                    <div><strong>Age:</strong> ${inputData.age} years</div>
                    <div><strong>BMI:</strong> ${inputData.bmi}</div>
                    <div><strong>Glucose:</strong> ${inputData.glucose} mg/dL</div>
                    <div><strong>Blood Pressure:</strong> ${inputData.blood_pressure} mmHg</div>
                </div>
            </div>
            <div class="result-item" style="margin-top: 15px;">
                <small><em>Note: This is a prediction based on machine learning models and should not replace professional medical advice.</em></small>
            </div>
        </div>
    `;
    
    container.className = `result-container ${resultClass} show`;
}

// Visualizations
async function loadVisualizations() {
    const vizContainer = document.getElementById('viz-container');
    
    // Show loading
    vizContainer.innerHTML = '<div class="spinner"></div><p style="text-align: center;">Loading visualizations...</p>';
    
    try {
        const response = await fetch('/api/visualizations');
        const data = await response.json();
        
        if (response.ok) {
            displayVisualizations(vizContainer, data);
        } else {
            vizContainer.innerHTML = `<div class="loading">Error loading visualizations: ${data.error}</div>`;
        }
    } catch (error) {
        console.error('Error:', error);
        vizContainer.innerHTML = '<div class="loading">Network error. Please try again.</div>';
    }
}

function displayVisualizations(container, data) {
    let visualizationsHTML = '';
    
    const vizTitles = {
        'age_distribution': 'Age Distribution',
        'bmi_glucose_scatter': 'BMI vs Glucose Scatter Plot',
        'outcome_pie': 'Diabetes Outcome Distribution',
        'correlation_heatmap': 'Feature Correlation Heatmap'
    };
    
    for (const [key, base64Image] of Object.entries(data)) {
        if (base64Image) {
            visualizationsHTML += `
                <div class="viz-item animate-fade-in">
                    <h3>${vizTitles[key] || key.replace('_', ' ').toUpperCase()}</h3>
                    <img src="data:image/png;base64,${base64Image}" alt="${vizTitles[key] || key}">
                </div>
            `;
        }
    }
    
    if (visualizationsHTML) {
        container.innerHTML = visualizationsHTML;
    } else {
        container.innerHTML = '<div class="loading">No visualizations available.</div>';
    }
}

// Utility functions
function showLoading(container, message) {
    container.innerHTML = `
        <div class="spinner"></div>
        <p style="text-align: center;">${message}</p>
    `;
    container.className = 'result-container show';
}

function showError(container, message) {
    container.innerHTML = `
        <div class="animate-fade-in">
            <h3><i class="fas fa-exclamation-triangle"></i> Error</h3>
            <p>${message}</p>
        </div>
    `;
    container.className = 'result-container result-danger show';
}

// Add some sample data buttons for quick testing
function fillSampleData(type) {
    if (type === 'diabetes-high-risk') {
        document.getElementById('pregnancies').value = 6;
        document.getElementById('glucose').value = 180;
        document.getElementById('blood-pressure').value = 95;
        document.getElementById('skin-thickness').value = 35;
        document.getElementById('insulin').value = 200;
        document.getElementById('bmi').value = 35.5;
        document.getElementById('diabetes-pedigree').value = 1.2;
        document.getElementById('age').value = 55;
    } else if (type === 'diabetes-low-risk') {
        document.getElementById('pregnancies').value = 1;
        document.getElementById('glucose').value = 95;
        document.getElementById('blood-pressure').value = 70;
        document.getElementById('skin-thickness').value = 20;
        document.getElementById('insulin').value = 80;
        document.getElementById('bmi').value = 22.5;
        document.getElementById('diabetes-pedigree').value = 0.3;
                document.getElementById('age').value = 25;
    }
}

function fillSampleText(type) {
    const textInput = document.getElementById('text-input');
    if (type === 'hate') {
        textInput.value = "I hate all people from that country, they are the worst!";
    } else if (type === 'normal') {
        textInput.value = "I love spending time with my family and friends. It's a beautiful day!";
    }
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + Enter to submit forms
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        if (currentTab === 'hate-speech') {
            predictHateSpeech();
        } else if (currentTab === 'diabetes') {
            predictDiabetes();
        }
    }
});

// Add real-time input validation
document.addEventListener('DOMContentLoaded', function() {
    // Add input event listeners for real-time validation
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('input', function() {
            validateNumericInput(this);
        });
    });
});

function validateNumericInput(input) {
    const value = parseFloat(input.value);
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    if (value < min || value > max) {
        input.style.borderColor = '#dc3545';
        input.style.backgroundColor = '#fff5f5';
    } else {
        input.style.borderColor = '#28a745';
        input.style.backgroundColor = '#f8fff8';
    }
}

