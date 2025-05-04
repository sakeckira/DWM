import pandas as pd
import numpy as np

# Dataset
data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                    'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
                   'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# Training
def train_naive_bayes(df, target):
    model = {}
    classes = df[target].unique()
    for cls in classes:
        subset = df[df[target] == cls]
        model[cls] = {}
        model[cls]['prior'] = len(subset) / len(df)
        for col in df.columns:
            if col != target:
                model[cls][col] = subset[col].value_counts(normalize=True).to_dict()
    return model

# Prediction
def predict_naive_bayes(model, instance):
    probs = {}
    for cls in model:
        prob = model[cls]['prior']
        for feature in instance:
            if feature in model[cls] and instance[feature] in model[cls][feature]:
                prob *= model[cls][feature][instance[feature]]
            else:
                prob *= 1e-6  # small value for unseen cases (Laplace smoothing idea)
        probs[cls] = prob
    return max(probs, key=probs.get)

# Train model
model = train_naive_bayes(df, 'PlayTennis')

# Predict
instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
prediction = predict_naive_bayes(model, instance)

print("Naive Bayes Model:\n", model)
print(f"\nPrediction for {instance}: {prediction}")
