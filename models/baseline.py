from models.header import *
from vocabulary import *

class Baseline:
    def __init__(self, keywords):
        self.keywords = keywords

    def forward(self, message):
        words = simplify_string(message)

        if [i for i in self.keywords if i in words]:
            return -1
        else:
            return 1

    def eval(self, data):
        validation_loss = 0
        spam_detected = 0
        spam_undetected = 0
        ham_detected = 0
        ham_undetected = 0

        for x, y in data:
            result = self.forward(x)

            if y == 1:
                if result >= 0:
                    ham_detected += 1
                else:
                    ham_undetected += 1
            else:
                if result < 0:
                    spam_detected += 1
                else:
                    spam_undetected += 1

            validation_loss += 10 * np.absolute(y - result)

        validation_loss /= len(data)

        return validation_loss, spam_detected, spam_undetected, ham_detected, ham_undetected

