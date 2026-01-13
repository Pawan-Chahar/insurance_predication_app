import pickle
import pandas as pd 


# import the ml model 
with open('model/model.pkl' , 'rb') as file:
    model = pickle.load(file)

# MLflow get model version

MODEL_VERSION = '1.0.0'

# Get class labels from model important for matching probablities to class name
class_labels = model.classes_.tolist()


def predict_output(user_input: dict):

    df = pd.DataFrame([user_input])

    predicted_class = model.predict(df)[0]
    #get the probaliblites of all classes
    probablities= model.predict_proba(df)[0]
    confidence = max(probablities)

    class_probs = dict(zip(class_labels, map(lambda p: round(p,4), probablities)))
    
    return {
        'predicted_category': predicted_class,
        'confidence': round(confidence, 4),
        'class_probablities': class_probs
    }
