import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
import base64
import io
from sklearn.utils import resample  # <-- Added for sampling

def plot_to_base64():
    """Convert matplotlib plot to base64 string"""
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

def create_visualizations():
    """Create sample visualizations for the dashboard"""
    
    # Sample data for diabetes analysis (imbalanced 70:30)
    np.random.seed(42)
    diabetes_data = pd.DataFrame({
        'Age': np.random.randint(20, 80, 100),
        'BMI': np.random.normal(28, 5, 100),
        'Glucose': np.random.normal(120, 30, 100),
        'Blood_Pressure': np.random.normal(80, 15, 100),
        'Outcome': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
    
    # Balance the dataset by undersampling the majority class
    majority_class = diabetes_data[diabetes_data.Outcome == 0]
    minority_class = diabetes_data[diabetes_data.Outcome == 1]
    
    majority_downsampled = resample(majority_class,
                                    replace=False,
                                    n_samples=len(minority_class),
                                    random_state=42)
    
    diabetes_data_balanced = pd.concat([majority_downsampled, minority_class])
    diabetes_data_balanced = diabetes_data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create visualizations
    visualizations = {}
    
    # 1. Age distribution
    plt.figure(figsize=(10, 6))
    plt.hist(diabetes_data_balanced['Age'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Age Distribution in Balanced Dataset', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    visualizations['age_distribution'] = plot_to_base64()
    
    # 2. BMI vs Glucose scatter plot
    plt.figure(figsize=(10, 6))
    colors = ['red' if x == 1 else 'blue' for x in diabetes_data_balanced['Outcome']]
    plt.scatter(diabetes_data_balanced['BMI'], diabetes_data_balanced['Glucose'], c=colors, alpha=0.6)
    plt.title('BMI vs Glucose Level (Balanced)', fontsize=16)
    plt.xlabel('BMI', fontsize=12)
    plt.ylabel('Glucose Level', fontsize=12)
    plt.grid(True, alpha=0.3)
    visualizations['bmi_glucose_scatter'] = plot_to_base64()
    
    # 3. Outcome distribution pie chart
    plt.figure(figsize=(8, 8))
    outcome_counts = diabetes_data_balanced['Outcome'].value_counts()
    plt.pie(outcome_counts.values, labels=['No Diabetes', 'Diabetes'], 
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    plt.title('Diabetes Outcome Distribution (Balanced)', fontsize=16)
    visualizations['outcome_pie'] = plot_to_base64()
    
    # 4. Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = diabetes_data_balanced.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap (Balanced)', fontsize=16)
    visualizations['correlation_heatmap'] = plot_to_base64()
    
    return visualizations
