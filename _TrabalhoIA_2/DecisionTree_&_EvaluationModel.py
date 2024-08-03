import pandas as pd
import numpy as np

from collections import Counter, defaultdict
from math import log2

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

choice = input("Choose the dataset: ")

dataset_paths = {
    'restaurant': r"restaurant.csv",
    'weather': r"weather.csv",
    'iris': r"iris.csv",
    'connect4': r"connect4_2.csv",
    'new': r"new.csv"                 # Novo ficheiro (alterar nome)          
}

def discretize_values(data):
    num_bins = int(np.log2(len(data)) + 1)
    bin_intervals = {}
    
    for column in data.columns[1:]:
        if pd.api.types.is_numeric_dtype(data[column]):
            discretized_values, bins = pd.cut(data[column], bins=num_bins, labels=False, retbins=True)
            data[column] = discretized_values
            bin_intervals[column] = bins
    
    return data, bin_intervals

if choice in dataset_paths:
    if choice == 'restaurant':
        csv_data = pd.read_csv('restaurant.csv', dtype={'Pat': str}, na_filter=False)
        csv_data['Pat'] = csv_data['Pat'].apply(lambda x: 'NONE' if x == 'None' else x)
        csv_data.to_csv('restaurant.csv', index=False)
    else:
        csv_data = pd.read_csv(dataset_paths[choice])
    headers = csv_data.columns.tolist()
    dataset = csv_data.values.tolist()
    csv_data, bin_intervals = discretize_values(csv_data)
    print("Dataset loaded and discretized successfully.")
    print("Bin intervals for discretized columns:")
    for column, bins in bin_intervals.items():
        print(f"{column}: {bins}")
else:
    print("Invalid choice. Please choose a valid dataset.")

def entropy(data):
    total = len(data)
    counts = Counter(row[-1] for row in data)
    return -sum((count/total) * log2(count/total) for count in counts.values())

def information_gain(data, attribute_index):
    total = len(data)
    subsets = defaultdict(list)
    
    for row in data:
        subsets[row[attribute_index]].append(row)
    
    weighted_entropy = sum((len(subset)/total) * entropy(subset) for subset in subsets.values())
    return entropy(data) - weighted_entropy

def best_attribute(data, attributes):
    gains = [(attr, information_gain(data, index)) for index, attr in enumerate(attributes)]
    return max(gains, key=lambda x: x[1])

print("Entropy of the dataset:", entropy(dataset))
print("Best attribute to split on:", best_attribute(dataset, headers[1:-1]))
print("- - - - - -")

class Node:
    def __init__(self, attribute=None, value=None, result=None, counter=None, branches=None):
        self.attribute = attribute
        self.value = value
        self.result = result
        self.counter = counter
        self.branches = branches if branches is not None else {}

def build_tree(data, attributes):
    if not data:
        return None
    
    class_counts = Counter(row[-1] for row in data)
    if len(class_counts) == 1:
        return Node(result=list(class_counts.keys())[0], counter=len(data))
    if not attributes:
        result = class_counts.most_common(1)[0][0]
        return Node(result=result, counter=len(data))
    best_attr, best_gain = best_attribute(data, attributes)
    best_attr_index = headers.index(best_attr)
    root = Node(attribute=best_attr)
    partitions = defaultdict(list)
    for row in data:
        partitions[row[best_attr_index]].append(row)
    
    for attr_value, subset in partitions.items():
        remaining_attrs = [attr for attr in attributes if attr != best_attr]
        if isinstance(attr_value, int):
            attr_value = str(attr_value)
        branch = build_tree(subset, remaining_attrs)
        root.branches[str(attr_value)] = branch
    return root

attributes = headers[1:-1]
decision_tree = build_tree(dataset, attributes)

def print_tree(node, indent=""):
    if node.result is not None:
        print(f"{indent}{node.result}: (Counter: {node.counter})")
    else:
        print(f"{indent}<{node.attribute}>")
        for value, branch in node.branches.items():
            print(f"{indent}{value}:")
            print_tree(branch, indent + "    ")

print_tree(decision_tree)

def PredictSingle(instance, decision_tree):
    if decision_tree.result is not None:
        return decision_tree.result
    
    attribute = decision_tree.attribute
    attr_value = instance.get(attribute)

    if (attr_value is None) or (attr_value not in decision_tree.branches):
        next
    else:
        return PredictSingle(instance, decision_tree.branches[attr_value])

def predict_file(input_file, decision_tree):
    predictions = []
    
    with open(input_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        instance = {}
        attributes = line.strip().split(',')
        for index, attr_value in enumerate(attributes):
            instance[headers[index+1]] = attr_value
        
        predicted_class = PredictSingle(instance, decision_tree)
        predictions.append(predicted_class)
    
    return predictions

print("- - - - - -")

if choice == 'restaurant':
    predictions = predict_file(r"test1.csv", decision_tree)
    print(f"Predicted classes: ", predictions)

elif choice == 'weather':
    predictions = predict_file(r"test2.csv", decision_tree)
    print(f"Predicted classes: ", predictions)

elif choice == 'iris':
    predictions = predict_file(r"test3.csv", decision_tree)
    print(f"Predicted classes: ", predictions)

elif choice == "connect4":
    instance = {
    'X1': 'b',
    'X2': 'b',
    'X3': 'b',
    'X4': 'b',
    'X5': 'b',
    'X6': 'b',
    'X7': 'b',
    'X8': 'b',
    'X9': 'b',
    'X10': 'b',
    'X11': 'b',
    'X12': 'b',
    'X13': 'x',
    'X14': 'o',
    'X15': 'b',
    'X16': 'b',
    'X17': 'b',
    'X18': 'b',
    'X19': 'x',
    'X20': 'o',
    'X21': 'x',
    'X22': 'o',
    'X23': 'x',
    'X24': 'o',
    'X25': 'b',
    'X26': 'b',
    'X27': 'b',
    'X28': 'b',
    'X29': 'b',
    'X30': 'b',
    'X31': 'b',
    'X32': 'b',
    'X33': 'b',
    'X34': 'b',
    'X35': 'b',
    'X36': 'b',
    'X37': 'b',
    'X38': 'b',
    'X39': 'b',
    'X40': 'b',
    'X41': 'b',
    'X42': 'b'
    }

elif choice == 'new':
    predictions = predict_file(r"test_new.csv", decision_tree) # Novo ficheiro de teste (alterar nome) 
    print(f"Predicted classes: ", predictions)


def evaluate_model(data, test_size=0.3):
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    train_tree = build_tree(train_data, headers[1:-1])
    
    test_instances = [{headers[i]: row[i] for i in range(1, len(headers) - 1)} for row in test_data]
    test_labels = [row[-1] for row in test_data]

    predictions = [PredictSingle(instance, train_tree) for instance in test_instances]
    
    test_labels = [str(label) for label in test_labels]
    predictions = [str(pred) for pred in predictions]

    print("Test labels: ", test_labels)
    print("Predictions: ", predictions)

    cm = confusion_matrix(test_labels, predictions)
    print("Confusion Matrix:")
    print(cm)

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(test_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='macro', zero_division=0)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

print("- - - - - -")

evaluate_model(dataset)