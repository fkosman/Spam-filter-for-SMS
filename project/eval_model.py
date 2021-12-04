from vocabulary import *
import sys
sys.path.append('../models')
from header import *
from lstm import LSTM
from baseline import Baseline

name = sys.argv[1]
model = LSTM(0,0,name, load=True)
baseline = Baseline(["FREE", "PRIZE", "AWARD"])

vocabulary = []
with open('../data/vocabulary.txt') as vocabfile:
    vocabfile.readline()
    vocabfile.readline()
    for line in vocabfile:
        line = line[:-1]
        vocabulary.append(line)

validation_encoded = []
validation_raw = []
with open('../data/validation_set.csv', encoding = "ISO-8859-1") as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        break
    for row in spamreader:
        if row[0] == "ham":
            validation_raw.append((row[1], 1))
            validation_encoded.append((encode_string(row[1], vocabulary), 1))
        else:
            validation_raw.append((row[1], -1))
            validation_encoded.append((encode_string(row[1], vocabulary), -1))

baseline_result = ""
model_result = ""

validation_loss, spam_detected, spam_undetected, ham_detected, ham_undetected = baseline.eval(validation_raw)

baseline_result += f"Loss: {validation_loss}\n"
baseline_result += "Accuracy: {:.2f}%\n".format(100 * (spam_detected + ham_detected) / 1000)
precisiion = spam_detected / (spam_detected + ham_undetected)
baseline_result += "Precision: {:.4f}\n".format(precisiion)
recall = spam_detected / 130
baseline_result += "Recall: {:.4f}\n".format(recall)
f_score = 1.25 * precisiion * recall / (0.25 * precisiion + recall)
baseline_result += "F_B score: {:.4f}\n".format(f_score)

validation_loss, spam_detected, spam_undetected, ham_detected, ham_undetected = model.eval(validation_encoded)

model_result += "Loss: {:.2f}\n".format(validation_loss[0,0])
model_result += "Accuracy: {:.2f}%\n".format(100 * (spam_detected + ham_detected) / 1000)
precisiion = spam_detected / (spam_detected + ham_undetected)
model_result += "Precision: {:.4f}\n".format(precisiion)
recall = spam_detected / 130
model_result += "Recall: {:.4f}\n".format(recall)
f_score = 1.25 * precisiion * recall / (0.25 * precisiion + recall)
model_result += "F_B score: {:.4f}\n".format(f_score)

print("\n\t-\tBaseline result")
print(baseline_result)
print("\n\t-\tModel result")
print(model_result)
