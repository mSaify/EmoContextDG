import pickle

import pandas as p


class Lang:
    def __init__(self, init_index2word):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word
        self.n_words = len(init_index2word)  # Count default tokens

    def index_words(self, sentence):
        print(sentence)
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

if __name__ == "__main__":

    print("LOADING empathetic_dialogue ...")
    with open('empdg_dataset_preproc.p', "rb") as f:
        [data_tra, data_val, data_tst, vocab] = pickle.load(f)

        print(data_tra)


        print('vocab')
        print(data_tst)