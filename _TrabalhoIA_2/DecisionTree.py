# import csv
import pandas as pd
import numpy as np

from collections import Counter, defaultdict
from math import log2

# dicionário para os paths dos ficheiros
# dataset_paths = {
#     'restaurant': r"C:\Users\User\Desktop\IA\2\restaurant.csv",
#     'weather': r"C:\Users\User\Desktop\IA\2\weather.csv",
#     'iris': r"C:\Users\User\Desktop\IA\2\iris.csv",
#     'connect4': r"C:\Users\User\Desktop\IA\2\connect4_2.csv"
# }

rows_destruidas = 0

def discValues(data):
    # Calcular o número de bins usando log2 do número de linhas mais um
    num_bins = int(np.log2(len(data)) + 1)
    bin_intervals = {}  # Dicionário para armazenar os intervalos dos bins
    
    for column in data.columns[1:]:
        if pd.api.types.is_numeric_dtype(data[column]):  # Apenas discretiza colunas numéricas
            # Discretiza os valores e obtém os bins
            discretized_values, bins = pd.cut(data[column], bins=num_bins, labels=False, retbins=True)
            data[column] = discretized_values
            bin_intervals[column] = bins
    
    return data, bin_intervals

dataset_paths = {
    'restaurant': r"restaurant.csv",
    'weather': r"weather.csv",
    'iris': r"iris.csv",
    'connect4': r"connect4_2.csv", # connect4 mas com a primeira linha
    'new': r"new.csv"              # alterar new para nome do ficheiro
}

while True:
    # Choose dataset
    choice = input("Choose the dataset: ")

    if choice in dataset_paths:
        csv_data = pd.read_csv(dataset_paths[choice])

        headers = csv_data.columns.tolist()
        dataset = csv_data.values.tolist()

        # Alteração (SoLongLondon.py --> LondonBoy.py)
        num_items = len(dataset[0])
        cleaned_dataset = []

        csv_data, bin_intervals = discValues(csv_data)
        
        for row in dataset:
            if len(row) == num_items:
                cleaned_dataset.append(row)
            else:
                rows_destruidas += 1

        print("Dataset loaded, cleaned, and discretized successfully.")
        print(f"Number of disregarded rows: {rows_destruidas}")
        print("- - - - - -")
        for column, bins in bin_intervals.items():
            print(f"{column}: {bins}")
        print("- - - - - -")
        break
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
# print("Entropy of the dataset:\n", entropy(dataset))
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
        # Int --> String
        if isinstance(attr_value, int):
            attr_value = str(attr_value)
        branch = build_tree(subset, remaining_attrs)
        root.branches[str(attr_value)] = branch  # -
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

# output_format = tree_to_output_format(decision_tree)
# for line in output_format:
#    print(line)

def PredictSingle(instance, decision_tree):
    if decision_tree.result is not None:
        return decision_tree.result
    
    attribute = decision_tree.attribute
    attr_value = instance.get(attribute)
    
    #passar para lista e int para str

    #list1 = list(decision_tree.branches.keys())

    #list2 = [str(x) for x in list1]

    #print(f"Attribute: {attribute}, Value: {attr_value}")

    if (attr_value is None) or (attr_value not in list(decision_tree.branches.keys())):
        #Follow the first branch
        next
    else:
        #Segue o filho com o atributo
        #print(f"Following branch for value: {attr_value}")
        return PredictSingle(instance, decision_tree.branches[attr_value])
    
def predict_file(input_file, decision_tree):
    predictions = []
    
    with open(input_file, 'r') as file:
        lines = file.readlines()

    for line in lines:
        instance = {}
        #assumir que estam separados por uma vírgula
        attributes = line.strip().split(',')
        for index, attr_value in enumerate(attributes):
            #saltar o id
            instance[headers[index+1]] = attr_value
        
        #print(instance)
        predicted_class = PredictSingle(instance, decision_tree)
        predictions.append(predicted_class)
    
    return predictions

# '''
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
    'X9': 'b',                #b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win
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
    predictions = PredictSingle(instance, decision_tree)
    print(f"Predicted classes: ", predictions)

elif choice == 'new':
    predictions = predict_file(r"test_new.csv", decision_tree)   #colocar teste para o ficheiro new
    print(f"Predicted classes: ", predictions)
# '''