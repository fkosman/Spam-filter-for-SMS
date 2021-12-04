from vocabulary import *
from models.header import *
from models.lstm import LSTM
import sys

name = sys.argv[1]
model = LSTM(0,0,name, load=True)

vocabulary = []
with open('data/vocabulary.txt') as vocabfile:
    vocabfile.readline()
    vocabfile.readline()
    for line in vocabfile:
        line = line[:-1]
        vocabulary.append(line)

finished = False

while not finished:
    message = input("\nEnter a sample message: ")

    _, _, _, _, _, _, _, _, outputs = model.forward(encode_string(message, vocabulary))

    if outputs[-1] >= 0:
        print("HAM")
    else:
        print("SPAM")

    print("\nTry again?")
    print("(1) Test another sample")
    print("(2) Exit")
    ans = int(input(""))

    while ans != 1 and ans != 2:
        print("Invalid choice.")
        print("\nTry again?")
        print("(1) Test another sample")
        print("(2) Exit")
        ans = int(input(""))

    if ans == 2: finished = True
