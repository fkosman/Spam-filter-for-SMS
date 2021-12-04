import sys
sys.path.append('../models')
from lstm import LSTM
import matplotlib.pyplot as plt

name = sys.argv[1]
model = LSTM(0,0,name, load=True)

training_loss = []
training_acc = []
validation_loss = []
validation_acc = []
total_epochs = 0
predicted_spam = []
correctly_predicted_spam = []

with open("../logs/" + model.name + ".log") as logfile:
    for line in logfile:
        total_epochs += 1
        logfile.readline()
        training_loss.append(float(logfile.readline().split()[2]))
        validation_loss.append(float(logfile.readline().split()[2]))
        training_acc.append(float(logfile.readline().split()[2][:-1]))
        validation_acc.append(float(logfile.readline().split()[2][:-1]))
        for i in range(2): logfile.readline()
        correct = float(logfile.readline().split()[4])
        incorrect = float(logfile.readline().split()[3])
        correctly_predicted_spam.append(correct)
        predicted_spam.append(correct + incorrect)
        logfile.readline()

plt.figure(1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Time')
plt.plot(range(1, total_epochs + 1), training_loss, label="Training loss")
plt.plot(range(1, total_epochs + 1), validation_loss, label="Validation loss")
plt.legend()

plt.figure(2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Time')
plt.plot(range(1, total_epochs + 1), training_acc, label="Training accuracy")
plt.plot(range(1, total_epochs + 1), validation_acc, label="Validation accuracy")
plt.legend()

plt.show()
