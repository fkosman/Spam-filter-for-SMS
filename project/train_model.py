from vocabulary import *
import sys
sys.path.append('../models')
from header import *
from lstm import LSTM
import csv

print("\nDo you want to start training a new model, or continue with an existing one?")
print("(1) Train new model")
print("(2) Load existing model")
ans = int(input(""))

while ans != 1 and ans != 2:
    print("Invalid choice.")
    print("\nDo you want to start training a new model, or continue with an existing one?")
    print("(1) Train new model")
    print("(2) Load existing model")
    ans = int(input(""))

if ans == 1:
    name = input("\nName the model\n(Leave blank and press 'Enter' to use default naming scheme): ")
    while len(name.split()) != 1:
        print("Name can't contain whitespace.")
        name = input("\nName the model\n(Leave blank and press 'Enter' to use default naming scheme): ")

    hidden_size = int(input("\nSelect the hidden layer size: "))
    while hidden_size < 1:
        print("Hidden layer size can't be less than 1.")
        hidden_size = int(input("\nSelect the hidden layer size: "))

    model = LSTM(hidden_size, vocab_size, name)

else:
    name = input("\nEnter the model name: ")
    model = LSTM(0,0,name, load=True)

    
lr = float(input("\nSelect the learning rate: "))
while lr < 0:
    print("Learning rate can't be less than 0.")
    lr = float(input("\nSelect the learning rate: "))

num_epochs = int(input("\nSelect the number of training epochs: "))
while num_epochs < 1:
    print("Number of epochs can't be less than 1.")
    num_epochs = int(input("\nSelect the number of epochs: "))


vocabulary = []
with open('../data/vocabulary.txt') as vocabfile:
    vocabfile.readline()
    vocabfile.readline()
    for line in vocabfile:
        line = line[:-1]
        vocabulary.append(line)

validation_set = []
with open('../data/validation_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        if row[0] == "ham":
            validation_set.append((encode_string(row[1], vocabulary), 1))
        else:
            validation_set.append((encode_string(row[1], vocabulary), -1))
            
training_set = []
with open('../data/training_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        if row[0] == "ham":
            training_set.append((encode_string(row[1], vocabulary), 1))
        else:
            training_set.append((encode_string(row[1], vocabulary), -1))

balanced_training_set = []
with open('../data/balanced_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        if row[0] == "ham":
            balanced_training_set.append((encode_string(row[1], vocabulary), 1))
        else:
            balanced_training_set.append((encode_string(row[1], vocabulary), -1))

print("\nDo you want to train with the complete training set, or the balanced one?")
print("(1) Full training dataset")
print("(2) Balanced training dataset")
ans = int(input(""))
while ans != 1 and ans != 2:
    print("Invalid choice.")
    print("\nDo you want to train with the complete training set, or the balanced one?")
    print("(1) Full training dataset")
    print("(2) Balanced training dataset")
    ans = int(input(""))

if ans == 1:
    model.train(training_set, validation_set, num_epochs, lr)
else:
    model.train(balanced_training_set, validation_set, num_epochs, lr)

print("\nEnd of training\n")
