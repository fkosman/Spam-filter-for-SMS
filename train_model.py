import sys
import csv
import random
from datetime import datetime
random.seed(datetime.now())
from models.header import *
from models.rnn import RNN
from models.lstm import LSTM

validation_set = []
with open('datasets/validation_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        entry = []
        for c in row[1]:
            entry.append(encode_char(c))

        if row[0] == "ham":
            validation_set.append((entry, 1))
        else:
            validation_set.append((entry, 0))

training_set = []
with open('datasets/training_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        entry = []
        for c in row[1]:
            entry.append(encode_char(c))

        if row[0] == "ham":
            training_set.append((entry, 1))
        else:
            training_set.append((entry, 0))

random.shuffle(training_set)
random.shuffle(training_set)

half_n_half = []
num_spam = 0
num_ham = 0
for item in training_set:
    if item[1] == 1 and num_ham < 500:
        half_n_half.append(item)
        num_ham += 1
    if item[1] == 0 and num_spam < 500:
        half_n_half.append(item)
        num_spam += 1
    if num_ham == 500 and num_spam == 500: break

if len(sys.argv) != 5:
    sys.exit("Invalid command line arguments.\n" +
             "Must enter: hidden-size, learning-rate, starting epoch, # of epochs")

hidden_size = sys.argv[1]
lr = sys.argv[2]
start_epoch = sys.argv[3]
num_epochs = sys.argv[4]

if hidden_size < 1 or lr < 1 or start_epoch < 1 or num_epochs < 1:
    sys.exit("Arguments must be positive numbers")

model = LSTM(hidden_size, vocab_size)

if start_epoch == 1:
    model.train(half_n_half, validation_set, start_epoch, num_epochs, lr)
else:
    model.load(half_n_half, validation_set, start_epoch, num_epochs, lr)
print("\nEnd of training\n")
