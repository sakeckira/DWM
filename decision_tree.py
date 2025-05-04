# import numpy as np
# import pandas as pd
# import math

# # Dataset
# df = pd.DataFrame({
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
# })

# # Entropy calculation
# def entropy(col):
#     vals, counts = np.unique(col, return_counts=True)
#     probs = counts / counts.sum()
#     return -np.sum(probs * np.log2(probs))

# # Information Gain
# def info_gain(data, feature, target='PlayTennis'):
#     total_entropy = entropy(data[target])
#     vals, counts = np.unique(data[feature], return_counts=True)
#     weighted = sum((counts[i]/counts.sum()) * entropy(data[data[feature]==vals[i]][target]) for i in range(len(vals)))
#     return total_entropy - weighted

# # ID3 Algorithm
# def ID3(data, features, target='PlayTennis'):
#     vals, counts = np.unique(data[target], return_counts=True)
#     if len(vals) == 1:
#         return vals[0]
#     if not features:
#         return vals[np.argmax(counts)]

#     gains = [info_gain(data, f, target) for f in features]
#     best = features[np.argmax(gains)]
#     tree = {best: {}}
    
#     for val in np.unique(data[best]):
#         subset = data[data[best] == val]
#         sub_features = [f for f in features if f != best]
#         tree[best][val] = ID3(subset, sub_features, target)
    
#     return tree

# # Prediction function
# def predict(tree, instance):
#     if not isinstance(tree, dict):
#         return tree  # Leaf node (final prediction)
    
#     feature = list(tree.keys())[0]
#     value = instance[feature]
    
#     if value in tree[feature]:
#         return predict(tree[feature][value], instance)
#     else:
#         return None  # In case the value is not found in the tree

# # Build tree
# features = list(df.columns[:-1])
# tree = ID3(df, features)

# # Example prediction (instance: 'Outlook' = 'Sunny', 'Temperature' = 'Hot', 'Humidity' = 'High', 'Wind' = 'Weak')
# instance = {'Outlook': 'Sunny', 'Temperature': 'Hot', 'Humidity': 'High', 'Wind': 'Weak'}
# prediction = predict(tree, instance)

# print("Generated Decision Tree:\n", tree)
# print(f"Prediction for {instance}: {prediction}")


import math
from collections import Counter
import pandas as pd

def calculate_entropy(data, target_column):
    class_counts = Counter([row[target_column] for row in data])
    total_rows = len(data)
    entropy = 0.0
    for count in class_counts.values():
        probability = count / total_rows
        entropy -= probability * math.log2(probability)
    return entropy

def calculate_information_gain(data, attribute, target_column):
    total_entropy = calculate_entropy(data, target_column)
    attribute_values = set([row[attribute] for row in data])
    weighted_entropy = 0.0
    total_rows = len(data)
    for value in attribute_values:
        subset = [row for row in data if row[attribute] == value]
        subset_size = len(subset)
        subset_entropy = calculate_entropy(subset, target_column)
        weighted_entropy += (subset_size / total_rows) * subset_entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_best_attribute(data, attributes, target_column):
    best_attribute = None
    max_gain = -1
    for attribute in attributes:
        gain = calculate_information_gain(data, attribute, target_column)
        print(f"Information Gain for {attribute}: {gain:.4f}")
        if gain > max_gain:
            max_gain = gain
            best_attribute = attribute
    return best_attribute

def build_decision_tree(data, attributes, target_column, tree=None):
    if tree is None:
        tree = {}
    classes = [row[target_column] for row in data]
    if len(set(classes)) == 1:
        return classes[0]
    if not attributes:
        majority_class = Counter(classes).most_common(1)[0][0]
        return majority_class
    best_attribute = find_best_attribute(data, attributes, target_column)
    print(f"\nBest attribute to split on: {best_attribute}\n")
    tree = {best_attribute: {}}
    remaining_attributes = [attr for attr in attributes if attr != best_attribute]
    attribute_values = set([row[best_attribute] for row in data])
    for value in attribute_values:
        subset = [row for row in data if row[best_attribute] == value]
        if not subset:
            majority_class = Counter(classes).most_common(1)[0][0]
            tree[best_attribute][value] = majority_class
        else:
            subtree = build_decision_tree(subset, remaining_attributes, target_column)
            tree[best_attribute][value] = subtree
    return tree

def classify(tree, instance):
    if isinstance(tree, str):
        return tree
    attribute = next(iter(tree))
    attribute_value = instance.get(attribute)
    if attribute_value not in tree[attribute]:
        return None
    subtree = tree[attribute][attribute_value]
    return classify(subtree, instance)

def print_rules(tree, rule=""):
    if isinstance(tree, str):
        print(f"IF {rule} THEN Class = {tree}")
        return
    attribute = next(iter(tree))
    for value, subtree in tree[attribute].items():
        new_rule = f"{rule} AND {attribute} = {value}" if rule else f"{attribute} = {value}"
        print_rules(subtree, new_rule)

def print_tree(tree, indent=""):
    if isinstance(tree, str):
        print(indent + "└── Class:", tree)
        return
    attribute = next(iter(tree))
    print(indent + attribute)
    for value, subtree in tree[attribute].items():
        print(indent + "├──", value)
        print_tree(subtree, indent + "│   ")

def load_csv_data(file_path, target_column):
    df = pd.read_csv(file_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")
    attributes = [col for col in df.columns if col != target_column]
    data = df.to_dict(orient='records')
    return data, attributes, target_column

def decision_tree_main():
    file_path = input("Enter the path to the CSV file: ").strip()
    df = pd.read_csv(file_path)
    print("\nAvailable columns:", ', '.join(df.columns))
    target_column = input("Enter the name of the target column: ").strip()
    data, attributes, target_column = load_csv_data(file_path, target_column)

    print("\nStep 1: Calculating Initial Entropy (Info(D))")
    initial_entropy = calculate_entropy(data, target_column)
    print(f"Initial Entropy (Info(D)): {initial_entropy:.4f}\n")

    print("Step 2: Calculating Information Gain for Each Attribute")
    for attribute in attributes:
        gain = calculate_information_gain(data, attribute, target_column)
        print(f"Information Gain for {attribute}: {gain:.4f}")

    print("\nStep 3: Building the Decision Tree")
    decision_tree = build_decision_tree(data, attributes, target_column)

    print("\nStep 4: Decision Tree Rules")
    print_rules(decision_tree)

    print("\nStep 5: Text-Based Tree Visualization")
    print_tree(decision_tree)

# Run the decision tree main function
decision_tree_main()
