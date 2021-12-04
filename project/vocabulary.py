import numpy as np
import csv
import operator

vocab_size = 1000
redundant_vocab_size = 60

def generate_vocab():
    spam_messages = []
    ham_messages = []

    with open('../data/training_set.csv', encoding = "ISO-8859-1") as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            break
        for row in spamreader:
            entry = []
            if row[0] == "ham":
                ham_messages.append(row[1])
            else:
                spam_messages.append(row[1])

    ham_vocab = [word for word, freq in most_common_words(ham_messages)]
    spam_vocab = [word for word, freq in most_common_words(spam_messages)]

    redundant_vocab = []

    for i in range(redundant_vocab_size):
        for j in range(redundant_vocab_size):
            if spam_vocab[i] == ham_vocab[j]:
                redundant_vocab.append(spam_vocab[i])

    for word in redundant_vocab:
        ham_vocab.remove(word)
        spam_vocab.remove(word)

    final_vocab = []

    i = 0
    j = 0
    turn = 0
    while len(final_vocab) < vocab_size - 1:
        if turn % 2 == 0:
            while ham_vocab[i] in final_vocab:
                i += 1
            final_vocab.append(ham_vocab[i])
            i += 1
        else:
            while spam_vocab[j] in final_vocab:
                j += 1
            final_vocab.append(spam_vocab[j])
            j += 1
        turn += 1

    final_vocab.append("[!EMPTY!]")
    return final_vocab

def most_common_words(messages):
    frequencies = {}
    vocabulary = []

    for message in messages:
        words = message.split()
        for wor in words:
            w = simplify_word(wor)
            if w not in vocabulary:
                vocabulary.append(w)
                frequencies[w] = 1
            else:
                frequencies[w] += 1

    most_common = dict(sorted(frequencies.items(), key=operator.itemgetter(1), reverse=True))
    return [item for item in most_common.items()]

def simplify_char(c):
    # Number of characters represented is 30
    # (26 letters + punctuation + numbers + white space + all others)
    char_num = ord(c)

    if char_num > 64 and char_num < 91:
        return c

    if char_num > 96 and char_num < 123:
        return chr(char_num - 32)

    if char_num > 47 and char_num < 58:
        return '5'

    if char_num < 33:
        return ' '

    if char_num > 32 and char_num < 128:
        return '!'

    return '@'

def simplify_word(w):
    result = ""
    for c in w:
        result += simplify_char(c)

    return result

def simplify_string(str):
    result = []
    for w in str.split():
        result.append(simplify_word(w))

    return result

def encode_word(w, vocab):
    one_hot_encoded = np.zeros((vocab_size, 1))
    one_hot_encoded[word_to_index(w, vocab), 0] = 1
    return one_hot_encoded

def encode_string(str, vocab):
    words = simplify_string(str)

    i = 0
    while i < len(words):
        if words[i] not in vocab:
            words.pop(i)
        else:
            i += 1

    encoded = []
    for item in words:
        encoded.append(encode_word(item, vocab))

    if len(encoded) == 0:
        encoded.append(encode_word(vocab[-1], vocab))

    return encoded

def decode_word(encoded, vocab):
    for i in range(vocab_size):
        if encoded[i, 0] == 1:
            return vocab[i]

    return vocab[-1]

def decode_string(inputs, vocab):
    str = ""
    for input in inputs:
        str += decode_word(input, vocab) + " "

    return str

def word_to_index(w, vocab):
    for i in range(vocab_size):
        if w == vocab[i]:
            return i

    return -1

with open("../data/vocabulary.txt", "w") as vocab_file:
    vocab = generate_vocab()
    vocab_file.write("Training vocabulary:\n")

    for word in vocab:
        vocab_file.write("\n" + word)

    vocab_file.write("\n")