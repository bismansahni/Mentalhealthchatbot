import joblib


model = joblib.load('models/faq_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict(question):
    
    question_vector = vectorizer.transform([question])
    
    prediction = model.predict(question_vector)

    return prediction

if __name__ == '__main__':
    
    new_question = input(str("Enter your question: "))
  
    
    predicted_answer = predict(new_question)
    print(f"Predicted Answer to the question {new_question} is : {predicted_answer}")
