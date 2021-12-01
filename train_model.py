import sys
import os.path
import csv
import random
from datetime import datetime
random.seed(datetime.now())
from vocabulary import *
from models.header import *
from models.rnn import RNN
from models.lstm import LSTM

if len(sys.argv) != 4:
    sys.exit("\nInvalid command line arguments.\n" +
             "Must enter: hidden-size, learning-rate, # of epochs\n")

hidden_size = int(sys.argv[1])
lr = float(sys.argv[2])
num_epochs = int(sys.argv[3])

if hidden_size < 1 or lr <= 0 or num_epochs < 1:
    sys.exit("Arguments must be positive numbers")

vocabulary = []
with open('data/vocabulary.txt') as vocabfile:
    vocabfile.readline()
    vocabfile.readline()
    for line in vocabfile:
        line = line[:-1]
        vocabulary.append(line)

validation_set = []
with open('data/validation_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        if row[0] == "ham":
            validation_set.append((encode_string(row[1], vocabulary), 1))
        else:
            validation_set.append((encode_string(row[1], vocabulary), -1))

"""
for (message, _), orig in zip(validation_set, original):
    print("\nOriginal message:")
    print(orig)
    print("\"Encoded\" message (what the model sees):")
    print(decode_string(message, vocabulary) + "\n")
"""
            
training_set = []
with open('data/training_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        if row[0] == "ham":
            training_set.append((encode_string(row[1], vocabulary), 1))
        else:
            training_set.append((encode_string(row[1], vocabulary), -1))

balanced_training_set = []
with open('data/balanced_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        if row[0] == "ham":
            balanced_training_set.append((encode_string(row[1], vocabulary), 1))
        else:
            balanced_training_set.append((encode_string(row[1], vocabulary), -1))

model = LSTM(hidden_size, vocab_size)

if os.path.isfile("saved/" + model.name + ".params"):
    model.load()

model.train(balanced_training_set, validation_set, num_epochs, lr)

print("\nEnd of training\n")
