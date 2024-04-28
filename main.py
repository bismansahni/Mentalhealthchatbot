import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib  

def load_data(filepath):
  
    return pd.read_csv(filepath)

def preprocess_data(data):
 
    
    
    return data['Questions']  

def train_model(X_train, y_train):
  
    model = LogisticRegression(max_iter=1000)  
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
  
    predictions = model.predict(X_test)
    

def save_model(model, filepath):
  
    joblib.dump(model, filepath)

def main():
    
    df = load_data('data/dataset.csv')
    
    
    questions = preprocess_data(df)  
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)
    y = df['Answers']  
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = train_model(X_train, y_train)
    
    
    evaluate_model(model, X_test, y_test)
    
    
    save_model(model, 'models/faq_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')

if __name__ == '__main__':
    main()
