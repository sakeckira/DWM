import numpy as np
import pandas as pd
import math

# Dataset
df = pd.DataFrame({
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
})

# Entropy calculation
def entropy(col):
    vals, counts = np.unique(col, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

# Information Gain
def info_gain(data, feature, target='PlayTennis'):
    total_entropy = entropy(data[target])
    vals, counts = np.unique(data[feature], return_counts=True)
    weighted = sum((counts[i]/counts.sum()) * entropy(data[data[feature]==vals[i]][target]) for i in range(len(vals)))
    return total_entropy - weighted

# ID3 Algorithm
def ID3(data, features, target='PlayTennis'):
    vals, counts = np.unique(data[target], return_counts=True)
    if len(vals) == 1:
        return vals[0]
    if not features:
        return vals[np.argmax(counts)]

    gains = [info_gain(data, f, target) for f in features]
    best = features[np.argmax(gains)]
    tree = {best: {}}
    
    for val in np.unique(data[best]):
        subset = data[data[best] == val]
        sub_features = [f for f in features if f != best]
        tree[best][val] = ID3(subset, sub_features, target)
    
    return tree

# Prediction function
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree  # Leaf node (final prediction)
    
    feature = list(tree.keys())[0]
    value = instance[feature]
    
    if value in tree[feature]:
        return predict(tree[feature][value], instance)
    else:
        return None  # In case the value is not found in the tree

# Build tree
features = list(df.columns[:-1])
tree = ID3(df, features)

# Example prediction (instance: 'Outlook' = 'Sunny', 'Temperature' = 'Hot', 'Humidity' = 'High', 'Wind' = 'Weak')
instance = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
prediction = predict(tree, instance)

print("Generated Decision Tree:\n", tree)
print(f"Prediction for {instance}: {prediction}")
