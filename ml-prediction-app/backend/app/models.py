import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
import re
import joblib
import os

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class HateSpeechDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression()
        self.is_trained = False
        self._create_sample_data()
        self.train()
    
    def _create_sample_data(self):
        # Sample hate speech data for demonstration
        hate_speech_data = [
    # Hateful / Offensive (label: )
    ("I hate all people from that country", 1),
    ("You are so stupid and ugly", 1),
    ("All people of that religion are terrorists", 1),
    ("Women should not have rights", 1),
    ("I hope you die", 1),
    ("You people are the worst", 1),
    ("Go back to where you came from", 1),
    ("You're just a parasite on society", 1),
    ("Those people are a disease", 1),
    ("They should be wiped off the planet", 1),
    ("Nobody wants your kind here", 1),
    ("You're a waste of oxygen", 1),
    ("They don't deserve to live", 1),
    ("You're the reason the world is messed up", 1),
    ("I want to punch you in the face", 1),
    ("You are garbage", 1),
    ("They ruin everything", 1),
    ("Get out of my country", 1),
    ("You're disgusting", 1),
    ("Your existence is a mistake", 1),
    ("People like you are trash", 1),
    ("You're not even human", 1),
    ("Stay away from me, freak", 1),
    ("You make me sick", 1),
    ("You’re nothing but a burden", 1),
    ("Burn in hell", 1),
    ("You are a disgrace", 1),
    ("You deserve the worst", 1),
    ("They are all criminals", 1),
    ("They should all be jailed", 1),
    ("You're a pathetic loser", 1),
    ("I wish you were never born", 1),
    ("You don’t belong here", 1),
    ("Just shut up forever", 1),
    ("No one likes your kind", 1),
    ("I’ll make sure you suffer", 1),
    ("You’ll always be a failure", 1),
    ("You're worthless", 1),
    ("They should be exterminated", 1),
    ("I despise people like you", 1),
    ("All of you are idiots", 1),
    ("No one wants you here", 1),
    ("You are not welcome", 1),
    ("You’re not allowed to exist", 1),
    ("We don’t need people like you", 1),
    ("You are the scum of the earth", 1),
    ("You're part of the problem", 1),
    ("You ruin everything you touch", 1),
    ("You're a danger to society", 1),
    ("You’re the enemy", 1),
    ("I love spending time with my family", 0),
    ("This is a beautiful day", 0),
    ("Thank you for your help", 0),
    ("I enjoy reading books", 0),
    ("The weather is nice today", 0),
    ("I appreciate your kindness", 0),
    ("Let's work together on this project", 0),
    ("I disagree with your opinion but respect it", 0),
    ("This movie was entertaining", 0),
    ("I'm looking forward to the weekend", 0),
    ("Happy birthday! Wishing you the best!", 0),
    ("Good luck on your exam", 0),
    ("Can you please help me with this?", 0),
    ("You're doing a great job", 0),
    ("I value your perspective", 0),
    ("Hope you're having a good day", 0),
    ("Let’s find a solution together", 0),
    ("That’s an interesting idea", 0),
    ("Let’s hear both sides", 0),
    ("You made a great point", 0),
    ("This dish tastes amazing", 0),
    ("You’ve improved a lot", 0),
    ("Thanks for your feedback", 0),
    ("Let’s schedule a meeting", 0),
    ("Your effort is appreciated", 0),
    ("This community is very helpful", 0),
    ("Glad we’re working as a team", 0),
    ("You're very talented", 0),
    ("Keep up the great work", 0),
    ("Your kindness matters", 0),
    ("That was very thoughtful", 0),
    ("I respect your decision", 0),
    ("You're an inspiration", 0),
    ("Let’s focus on learning", 0),
    ("We all make mistakes, it’s okay", 0),
    ("Constructive criticism is welcome", 0),
    ("Let’s move forward positively", 0),
    ("That was a helpful tip", 0),
    ("Your honesty is appreciated", 0),
    ("I admire your dedication", 0),
    ("You bring great energy", 0),
    ("We’re lucky to have you here", 0),
    ("Everyone deserves respect", 0),
    ("Differences make us stronger", 0),
    ("This team has great potential", 0),
    ("Thanks for always being supportive", 0),
    ("We should always show empathy", 0),
    ("That’s a very kind gesture", 0),
    ("Gratitude goes a long way", 0),
    ("Learning is a lifelong journey", 0),
    ("Let me know if you need anything", 0),
    ("Hope you have a wonderful day", 0),
    ("Thanks for sharing your thoughts", 0),
    ("I'm open to new ideas", 0),
    ("That’s an interesting perspective", 0),
    ("Nice work on that assignment", 0),
    ("I'm here if you want to talk", 0),
    ("Your contribution was valuable", 0),
    ("It’s okay to ask questions", 0),
    ("Let’s take a break and come back fresh", 0),
    ("Good morning, how are you?", 0),
    ("Please let me know your opinion", 0),
    ("We should consider all viewpoints", 0),
    ("Every idea counts", 0),
    ("Feel free to express yourself", 0),
    ("Let’s support each other", 0),
    ("This looks promising", 0),
    ("Collaboration makes us stronger", 0),
    ("Let’s try to understand each other", 0),
    ("You’ve done well so far", 0),
    ("No worries, take your time", 0),
    ("That’s perfectly understandable", 0),
    ("Let’s take this one step at a time", 0),
    ("You’re not alone in this", 0),
    ("Thanks for your patience", 0),
    ("You handled that very well", 0),
    ("I admire your honesty", 0),
    ("It's okay to feel that way", 0),
    ("Let's meet halfway", 0),
    ("Your input is always appreciated", 0),
    ("I’m happy to help anytime", 0),
    ("Everyone is learning at their own pace", 0),
    ("We all have something to offer", 0),
    ("This is a great opportunity to grow", 0),
    ("That’s a fair assessment", 0),
    ("Don’t hesitate to reach out", 0),
    ("We all have different strengths", 0),
    ("You're capable of amazing things", 0),
    ("We should listen before judging", 0),
    ("Learning from mistakes is important", 0),
    ("Keep striving for excellence", 0),
    ("Thanks for helping out", 0),
    ("Let’s try a different approach", 0),
    ("You’ve come a long way", 0),
    ("Teamwork makes everything better", 0),
    ("You did the best you could", 0),
    ("You’re doing fine, just keep going", 0),
    ("Appreciate the small wins", 0),
    ("Everyone starts somewhere", 0),
    ("The journey is just as important as the goal", 0),
    ("That’s a solid effort", 0),
    ("Mistakes help us grow", 0),
    ("Thanks for being understanding", 0),
    ("I’m learning from you", 0),
    ("Keep being curious", 0),
    ("You’re making progress", 0),
    ("This is a great discussion", 0),
    ("It’s okay to take a break", 0),
    ("We should be mindful of others", 0),
    ("Respect goes a long way", 0),
    ("We all have unique experiences", 0),
    ("Let’s be patient with one another", 0),
    ("This was a productive meeting", 0),
    ("Let’s focus on solutions", 0),
    ("The effort you put in shows", 0),
    ("It’s never too late to improve", 0),
    ("We’re all in this together", 0),
    ("Kindness can change everything", 0),
    ("Every voice matters", 0),
    ("That was a thoughtful response", 0),
    ("Empathy makes a difference", 0),
    ("It’s great to see your growth", 0),
    ("Your insight is meaningful", 0),
    ("It’s inspiring to work with you", 0),
    ("You have great communication skills", 0),
    ("This community is amazing", 0),
    ("Let's keep learning together", 0),
    ("I appreciate your attention to detail", 0),
    ("You’re setting a great example", 0),
    ("It’s okay to slow down and reflect", 0),
    ("You’re making a positive impact", 0),
    ("Your presence makes a difference", 0),
    ("Let’s continue to support each other", 0),
    ("Thanks for being respectful", 0),
    ("We can always learn from others", 0),
    ("You're making a good point", 0),
    ("Every day is a chance to improve", 0),
    ("Let’s try to be more inclusive", 0),
    ("It’s okay to change your mind", 0),
    ("You're capable of so much", 0),
    ("Small steps lead to big changes", 0),
    ("You bring something special", 0),
    ("Thank you for showing up today", 0),
    ("Keep asking good questions", 0),
    ("You're a great listener", 0),
    ("We’re here to grow, not to judge", 0),
    ("That was a kind thing to say", 0),
    ("We all have bad days — it’s okay", 0),
    ("You bring a fresh perspective", 0),
    ("Thanks for taking the time", 0),
    ("Let’s continue improving together", 0),
    ("It’s okay to make adjustments", 0),
    ("We're all working toward better", 0),
    ("Keep showing up — it matters", 0),
    ("You don't deserve to be here", 1),
    ("That group is nothing but trouble", 1),
    ("You people are ruining this place", 1),
    ("Why don’t you just disappear?", 1),
    ("You're always in the way", 1),
    ("You’re a complete failure", 1),
    ("They should lock you all up", 1),
    ("No one cares about you", 1),
    ("I can’t stand your kind", 1),
    ("Go rot somewhere", 1),
    ("You’re a joke", 1),
    ("You make everything worse", 1),
    ("You belong in the trash", 1),
    ("You don’t matter", 1),
    ("Everything you say is nonsense", 1),
    ("You’re just a problem", 1),
    ("Get lost already", 1),
    ("Why are you even talking?", 1),
    ("You're disgusting to look at", 1),
    ("Nobody wants to hear from you", 1),
    ("You don’t fit in", 1),
    ("You're always ruining things", 1),
    ("You’re the dumbest person I know", 1),
    ("You have no worth", 1),
    ("Go away forever", 1),
    ("You're a total embarrassment", 1),
    ("How can you live with yourself?", 1),
    ("Everything about you is wrong", 1),
    ("You're just noise", 1),
    ("Your opinion is garbage", 1),
    ("You're the reason this place is awful", 1),
    ("I can't believe how useless you are", 1),
    ("You’re the worst kind of person", 1),
    ("Nobody wants people like you", 1),
    ("You're so messed up", 1),
    ("You're not welcome here", 1),
    ("You're a burden to everyone", 1),
    ("Go back to your hole", 1),
    ("You're a freak", 1),
    ("Just leave and never return", 1),
    ("Your voice is annoying", 1),
    ("Stop acting like you matter", 1),
    ("No one would care if you disappeared", 1),
    ("You're a nobody", 1),
    ("You ruin everything you touch", 1),
    ("You’re absolutely pathetic", 1),
    ("Nobody needs you", 1),
    ("You're just a disaster", 1),
    ("You make things worse by existing", 1),
    ("You're always the problem", 1),
    ("Just shut up already", 1),
    ("You're pointless", 1),
    ("They shouldn't allow people like you", 1),
    ("You’ll never amount to anything", 1),
    ("You make me angry just by being here", 1),
    ("You should be ashamed to exist", 1),
    ("You mess up everything", 1),
    ("You're such a disappointment", 1),
    ("No one needs to hear from you", 1),
    ("You’re a waste of time", 1),
    ("You’re always making things worse", 1),
    ("Your existence is painful", 1),
    ("You contribute nothing", 1),
    ("Everything you do is a mistake", 1),
    ("You belong in a cage", 1),
    ("You don't deserve kindness", 1),
    ("You're only good at being awful", 1),
    ("You’re beneath everyone else", 1),
    ("You're a black mark on society", 1),
    ("You should be removed", 1),
    ("You don’t have the right to speak", 1),
    ("You're scum", 1),
    ("You're always dragging people down", 1),
    ("Nobody believes in you", 1),
    ("You’re the lowest of the low", 1),
    ("You're always in the way", 1),
    ("You’re a walking mistake", 1),
    ("You're a disgrace to this world", 1),
    ("You can't do anything right", 1),
    ("Everyone hates you", 1),
    ("You're just background noise", 1),
    ("You're poison", 1),
    ("You belong nowhere", 1),
    ("Nobody wants your presence", 1),
    ("You're a total waste", 1),
    ("You ruin every moment", 1),
    ("Why are you even here?", 1),
    ("Your kind brings nothing good", 1),
    ("You're pathetic beyond words", 1),
    ("You're just a curse", 1),
    ("Even silence is better than you", 1),
    ("The world would be better without you", 1),
    ("You're a joke no one laughs at", 1),
    ("You're only here to annoy people", 1),
    ("Everything about you is irritating", 1),
    ("You're like a virus", 1),
    ("You're always the weakest link", 1),
    ("You're the reason people quit", 1),
    ("You're never going to change", 1),
    ("You’re the worst decision ever made", 1),
    ("Even failure is better than you", 1),
    ("You're a walking disaster", 1),
    ("People avoid you for a reason", 1),
    ("You make life harder for everyone", 1)
]

        
        # Expand the dataset
        expanded_data = hate_speech_data * 10  # Repeat for more training data
        
        texts, labels = zip(*expanded_data)
        self.X = list(texts)
        self.y = list(labels)
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def train(self):
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in self.X]
        
        # Vectorize texts
        X_vectorized = self.vectorizer.fit_transform(processed_texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, self.y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Hate Speech Model Accuracy: {accuracy:.2f}")
        
        self.is_trained = True
    
    def predict(self, text):
        if not self.is_trained:
            return "Model not trained"
        
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(vectorized_text)[0]
        
        return "Hate Speech" if prediction == 1 else "Normal Speech"
    
    def get_confidence(self, text):
        if not self.is_trained:
            return 0.0
        
        processed_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([processed_text])
        probabilities = self.model.predict_proba(vectorized_text)[0]
        
        return max(probabilities)

class DiabetesPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self._create_sample_data()
        self.train()
    
    def _create_sample_data(self):
        # Create synthetic diabetes dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        pregnancies = np.random.randint(0, 15, n_samples)
        glucose = np.random.normal(120, 30, n_samples)
        blood_pressure = np.random.normal(80, 15, n_samples)
        skin_thickness = np.random.normal(25, 10, n_samples)
        insulin = np.random.normal(100, 50, n_samples)
        bmi = np.random.normal(28, 8, n_samples)
        diabetes_pedigree = np.random.uniform(0.1, 2.0, n_samples)
        age = np.random.randint(18, 80, n_samples)
        
        # Create target based on realistic conditions
        diabetes_risk = (
            (glucose > 140) * 0.3 +
            (bmi > 30) * 0.2 +
            (age > 45) * 0.2 +
            (diabetes_pedigree > 0.5) * 0.15 +
            (blood_pressure > 90) * 0.1 +
            (pregnancies > 5) * 0.05
        )
        
        outcome = (diabetes_risk + np.random.normal(0, 0.1, n_samples)) > 0.4
        
        self.X = np.column_stack([
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, diabetes_pedigree, age
        ])
        self.y = outcome.astype(int)
    
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Diabetes Model Accuracy: {accuracy:.2f}")
        
        self.is_trained = True
    
    def predict(self, features):
        if not self.is_trained:
            return 0
        
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)[0]
        
        return prediction
    
    def get_probability(self, features):
        if not self.is_trained:
            return 0.0
        
        features_array = np.array(features).reshape(1, -1)
        probability = self.model.predict_proba(features_array)[0][1]
        
        return probability
