import random
from datetime import datetime
random.seed(datetime.now())
import csv

raw_data = []
with open('datasets/spam.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        raw_data.append(row)

random.shuffle(raw_data)
random.shuffle(raw_data)

num_spam = 0
num_ham = 0
with open("datasets/validation_set.csv", "w") as validation_file, open("datasets/training_set.csv", "w") as training_file:
    validation_file.write("v1,v2,,,\n")
    training_file.write("v1,v2,,,\n")

    validation_writer = csv.writer(validation_file)
    training_writer = csv.writer(training_file)
    for row in raw_data:
        if num_ham < 870 and row[0] == "ham":
            validation_writer.writerow(row)
            num_ham += 1
            continue
        if num_spam < 130 and row[0] == "spam":
            validation_writer.writerow(row)
            num_spam += 1
            continue

        training_writer.writerow(row)
