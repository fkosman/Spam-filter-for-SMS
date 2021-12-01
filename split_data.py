import random
from datetime import datetime
random.seed(datetime.now())
import csv

raw_data = []
with open('data/spam.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        raw_data.append(row)

random.shuffle(raw_data)
random.shuffle(raw_data)

num_spam = 0
num_ham = 0
training_data = []
with open("data/validation_set.csv", "w") as validation_file:
    validation_file.write("v1,v2,,,\n")

    validation_writer = csv.writer(validation_file)
    for row in raw_data:
        if num_ham < 870 and row[0] == "ham":
            validation_writer.writerow(row)
            num_ham += 1
            continue
        if num_spam < 130 and row[0] == "spam":
            validation_writer.writerow(row)
            num_spam += 1
            continue

        training_data.append(row)

random.shuffle(training_data)
random.shuffle(training_data)

with open("data/training_set.csv", "w") as training_file:
    training_file.write("v1,v2,,,\n")

    training_writer = csv.writer(training_file)
    for row in training_data:
        training_writer.writerow(row)

num_spam = 0
balanced_data = []
for row in training_data:
    if row[0] == "spam":
        balanced_data.append(row)
        num_spam += 1

num_ham = 0
for row in training_data:
    if num_ham == num_spam:
        break
    if row[0] == "ham":
        balanced_data.append(row)
        num_ham += 1

random.shuffle(balanced_data)
random.shuffle(balanced_data)

with open("data/balanced_set.csv", "w") as balanced_file:
    balanced_file.write("v1,v2,,,\n")

    balanced_writer = csv.writer(balanced_file)
    for row in balanced_data:
        balanced_writer.writerow(row)

