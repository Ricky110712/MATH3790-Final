import sys
import numpy as np
import codecs
import jieba
import matplotlib.pyplot as plt

if sys.version_info.major == 3:
    xrange = range


class BaselineSeg:
    def __init__(self, file_path):
        self.dic = {}
        self.read_dict(file_path)

    # Read dictionary
    def read_dict(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                items = line.split(',')
                self.dic[items[0]] = True

    # Forward maximum matching segmentation
    def max_match_seq(self, string):
        st = 0
        ed = len(string)
        lst = []
        while st != ed:
            word = ''
            t = st
            while t != ed:
                temp = string[st: t + 1]
                if temp in self.dic:
                    word = temp
                t += 1
            if not word:
                word = string[st: st + 1]
            lst.append(word)
            st += len(word)
        return lst


class HMMSeg:
    def __init__(self):
        # Hidden states: 0: word start, 1: word middle, 2: word end, 3: single character word
        self.a = np.zeros([4, 4])
        self.pi = np.zeros([4])
        self.dic = {}

    # Get hidden state sequence from word sequence
    def get_hidden_seq(self, word_seq):
        seq = []
        for word in word_seq:
            wsiz = len(word)
            if wsiz == 1:
                seq.append(3)
            else:
                for i in range(wsiz):
                    if i == 0:
                        seq.append(0)
                    elif i != wsiz - 1:
                        seq.append(1)
                    else:
                        seq.append(2)
        return np.array(seq, dtype=np.int32)

    # Convert sentence to observed sequence
    def get_observed_seq(self, sentence):
        siz = len(sentence)
        o = np.zeros([siz], dtype=np.int32)
        for i in range(siz):
            o[i] = self.dic[sentence[i]] if sentence[i] in self.dic else 0
        return o

    # Load dictionary for each character
    def load_word_dic(self, file_path):
        with codecs.open(file_path, 'r', 'utf-8') as f:
            code = 1
            for line in f:
                word_seq = line.split()
                for word in word_seq:
                    for char in word:
                        if char not in self.dic:
                            self.dic[char] = code
                            code += 1

    # Viterbi algorithm
    def HMMViterbi(self, o):
        N = np.shape(self.b)[0]
        T = np.shape(o)[0]

        path = np.zeros(T, dtype=np.int32)
        delta = np.zeros((N, T))
        phi = np.zeros((N, T), dtype=np.int32)

        delta[:, 0] = self.pi * self.b[:, o[0]]
        for t in range(1, T):
            for i in range(N):
                tmp = delta[:, t - 1] * self.a[:, i]
                delta[i, t] = np.max(tmp) * self.b[i, o[t]]
                phi[i, t] = np.argmax(tmp)

        # Force end with state 2 or 3
        path[T - 1] = 2 if delta[2, T - 1] > delta[3, T - 1] else 3
        for t in range(T - 1, 0, -1):
            path[t - 1] = phi[path[t], t]
        return path

    # Segmentation
    def seg(self, sentence):
        o = self.get_observed_seq(sentence)
        path = self.HMMViterbi(o)
        siz = len(path)
        res = []
        word = ''
        for i in range(siz):
            word += sentence[i]
            if path[i] >= 2:
                res.append(word)
                word = ''
        return res

    # Training
    def fit(self, file_path):
        # Get encoding for each character
        self.load_word_dic(file_path)
        # Calculate a, b, pi
        self.b = np.zeros([4, len(self.dic.keys()) + 1])
        with codecs.open(file_path, 'r', 'utf-8') as f:
            for line in f:
                word_seq = line.split()
                if not word_seq:
                    continue
                hidden_seq = self.get_hidden_seq(word_seq)
                idx = 0
                for word in word_seq:
                    for char in word:
                        if idx == 0:
                            self.pi[hidden_seq[idx]] += 1
                        else:
                            self.a[hidden_seq[idx - 1], hidden_seq[idx]] += 1
                        self.b[hidden_seq[idx], self.dic[char]] += 1
                        idx += 1
        self.pi /= self.pi.sum()
        for i in range(4):
            self.a[i] /= self.a[i].sum()
            self.b[i] /= self.b[i].sum()
        # Set unknown characters as single character words
        self.b[3, 0] = 1


# Calculate similarity between two segmentation results
def similarity(a, b):
    lena = len(a) + 1
    lenb = len(b) + 1
    dp = np.zeros([lena, lenb])
    for i in range(1, lena):
        for j in range(1, lenb):
            if a[i - 1] == b[j - 1]:
                dp[i, j] = dp[i - 1, j - 1] + 1
            else:
                dp[i, j] = max(dp[i - 1, j], dp[i, j - 1])
    return 2 * dp[lena - 1, lenb - 1] / (lena + lenb - 2)


# Testing and plotting
def test_and_plot(test_file, groundtruth_file, baseline, hmm):
    with codecs.open(test_file, 'r', 'utf-8') as f1, codecs.open(groundtruth_file, 'r', 'utf-8') as f2:
        tot = 0
        yes_bas = [0, 0, 0, 0]  # sim >= 0.9, >= 0.8, >= 0.7, >= 0.6
        yes_hmm = [0, 0, 0, 0]  # sim >= 0.9, >= 0.8, >= 0.7, >= 0.6
        yes_jie = [0, 0, 0, 0]  # sim >= 0.9, >= 0.8, >= 0.7, >= 0.6
        for line1, line2 in zip(f1, f2):
            line1 = ''.join(line1.split())
            groundtruth_word_seq = line2.split()
            for method, yes in zip([baseline.max_match_seq, hmm.seg, jieba.cut], [yes_bas, yes_hmm, yes_jie]):
                word_seq = method(line1)
                if method == jieba.cut:
                    word_seq = list(word_seq)
                sim = similarity(word_seq, groundtruth_word_seq)
                for i, threshold in enumerate([0.9, 0.8, 0.7, 0.6]):
                    if sim >= threshold:
                        yes[i] += 1
            tot += 1

        # Calculate percentages
        percentages = {
            'baseline': [yes * 100 / tot for yes in yes_bas],
            'hmm': [yes * 100 / tot for yes in yes_hmm],
            'jieba': [yes * 100 / tot for yes in yes_jie]
        }

        # Print results to console
        print("Similarity percentages for each segmentation method:")
        for method in percentages:
            print(f"{method.capitalize()} Segmentation:")
            for threshold, percentage in zip(['>=90%', '>=80%', '>=70%', '>=60%'], percentages[method]):
                print(f"  {threshold}: {percentage:.2f}%")
            print()

        # Plotting
        thresholds = ['>=90%', '>=80%', '>=70%', '>=60%']
        x = np.arange(len(thresholds))
        width = 0.2

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, percentages['baseline'], width, label='Baseline')
        rects2 = ax.bar(x, percentages['hmm'], width, label='HMM')
        rects3 = ax.bar(x + width, percentages['jieba'], width, label='Jieba')

        ax.set_xlabel('Similarity Threshold')
        ax.set_ylabel('Percentage')
        ax.set_title('Comparison of Segmentation Methods by Similarity Threshold')
        ax.set_xticks(x)
        ax.set_xticklabels(thresholds)
        ax.legend()

        fig.tight_layout()

        plt.show()



if __name__ == '__main__':
    baseline = BaselineSeg('./DataSet/dict1.txt')
    hmm = HMMSeg()
    hmm.fit('./DataSet/msr_training.utf8')
    test_and_plot('./DataSet/msr_test.utf8', './DataSet/msr_test_gold.utf8', baseline, hmm)
