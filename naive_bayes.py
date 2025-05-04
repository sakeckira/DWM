# import pandas as pd
# import numpy as np

# # Dataset
# data = {
#     'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
#                 'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
#     'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
#                     'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
#     'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
#                  'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
#     'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
#              'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
#     'PlayTennis': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
#                    'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
# }
# df = pd.DataFrame(data)

# # Training
# def train_naive_bayes(df, target):
#     model = {}
#     classes = df[target].unique()
#     for cls in classes:
#         subset = df[df[target] == cls]
#         model[cls] = {}
#         model[cls]['prior'] = len(subset) / len(df)
#         for col in df.columns:
#             if col != target:
#                 model[cls][col] = subset[col].value_counts(normalize=True).to_dict()
#     return model

# # Prediction
# def predict_naive_bayes(model, instance):
#     probs = {}
#     for cls in model:
#         prob = model[cls]['prior']
#         for feature in instance:
#             if feature in model[cls] and instance[feature] in model[cls][feature]:
#                 prob *= model[cls][feature][instance[feature]]
#             else:
#                 prob *= 1e-6  # small value for unseen cases (Laplace smoothing idea)
#         probs[cls] = prob
#     return max(probs, key=probs.get)

# # Train model
# model = train_naive_bayes(df, 'PlayTennis')

# # Predict
# instance = {'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'}
# prediction = predict_naive_bayes(model, instance)

# print("Naive Bayes Model:\n", model)
# print(f"\nPrediction for {instance}: {prediction}")


import pandas as pd

#For excel file. Jaha bhi tumhara file stored hai, uska location
df = pd.read_csv(r'/content/plays_football.csv')


# Assume last column is the class (target)
target_col = df.columns[-1]



# Get feature columns (excluding the target column)
feature_cols = [col for col in df.columns if col != target_col]

# Step 1: Prior probabilities
total_samples = len(df)
classes = df[target_col].unique()
priors = {c: len(df[df[target_col] == c]) / total_samples for c in classes}

print("\nPrior Probabilities:")
for c in priors:
    print(f"P({target_col} = '{c}') = {round(priors[c], 3)}")

# Step 2: Likelihoods
def calc_likelihood(df, feature, feature_value, target_class):
    subset = df[df[target_col] == target_class]
    feature_count = len(subset[subset[feature] == feature_value])
    total_count = len(subset)
    return feature_count / total_count if total_count > 0 else 0

# Let's say this is the input sample to predict:
# (Change these values based on actual test case)
sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Windy': 'Strong'
}

# Step 3: Calculate likelihoods and posteriors
likelihoods = {}
posteriors = {}

print("\nLikelihoods:")

for c in classes:
    prob = 1
    print(f"\nFor class = '{c}':")
    for feature in feature_cols:
        feature_val = sample.get(feature)
        likelihood = calc_likelihood(df, feature, feature_val, c)
        prob *= likelihood
        print(f"P({feature} = '{feature_val}' | {target_col} = '{c}') = {round(likelihood, 3)}")
    posteriors[c] = prob * priors[c]

# Step 4: Print posterior probabilities
print("\nPosterior Probabilities (after applying Bayes' theorem):")
for c in posteriors:
    print(f"P(X | {target_col} = '{c}') * P({target_col} = '{c}') = {round(posteriors[c], 3)}")

# Step 5: Predict the class
prediction = max(posteriors, key=posteriors.get)
print(f"\nPredicted class: {target_col} = '{prediction}'")


