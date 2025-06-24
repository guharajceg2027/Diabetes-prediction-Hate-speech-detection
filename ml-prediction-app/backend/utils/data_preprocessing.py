import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        if self.stop_words:
            words = word_tokenize(text)
            text = ' '.join([word for word in words if word not in self.stop_words])
        
        return text
    
    def clean_numerical_data(self, df):
        """Clean and preprocess numerical data"""
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                # Fill numerical columns with median
                df[column] = df[column].fillna(df[column].median())
            else:
                # Fill categorical columns with mode
                df[column] = df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown')
        
        return df
    
    def scale_features(self, X):
        """Scale numerical features"""
        return self.scaler.fit_transform(X)
    
    def encode_labels(self, y):
        """Encode categorical labels"""
        return self.label_encoder.fit_transform(y)
    
    def detect_outliers(self, df, columns=None):
        """Detect outliers using IQR method"""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers = {}
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
        
        return outliers
