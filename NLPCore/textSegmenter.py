import re
from functools import lru_cache
from math import log10
import json
import os
from typing import Union, Dict

from utils import safely_load_json

"""
The Segmenter Class implements the Viterbi algorithm for word segmentation.
Based on CH14 from the book Beautiful Data (Segaran and Hammerbacher, 2009)
"""

REGEX_TOKEN = re.compile(r'\b[a-z]{2,}\b')
NGRAM_SEP = "_"
FILE_EXPRESSIONS = __file__[:__file__.rfind("/")] + "/../NLPCorpi/spellCheckerRegularExprs.txt"
if not os.path.exists(FILE_EXPRESSIONS):
    FILE_EXPRESSIONS = "./nlprocessors/functionalities/sharedFunctionalities/NLPCorpi/spellCheckerRegularExprs.txt"


class Pdist(dict):
    """
    A probability distribution estimated from word counts
    Notice: if pw = Pdist(unigrams, n_tokens:
        * pw[w] is the raw count of the word w
        * pw(w) is the probability of the word w
    """

    @staticmethod
    def default_unk_func(key, total):
        return 1. / total

    def __init__(self, data=None, total=None, unk_func=None, **kwargs):
        super().__init__(**kwargs)

        # insert the word counts
        data = data or {}
        for key, count in data.items():
            self[key] = self.get(key, 0) + int(count)

        self.total = float(total or sum(self.values()))
        self.unk_prob = unk_func or self.default_unk_func

    def __call__(self, key):
        if key in self:
            return self[key] / self.total
        else:
            return self.unk_prob(key, self.total)


class Segmenter:
    def __init__(self, corpus1gram: Union[Dict, str], corpus2gram: Union[Dict, str], max_split_length=20):
        """
        Args:
            corpus1gram (str): the statistics from which corpus to use for
                the spell correction.
            max_split_length (int): the maximum length of that a word can have
                for looking for splits
        """
        if isinstance(corpus1gram, dict) and isinstance(corpus2gram, dict):
            self.unigrams = corpus1gram
            self.bigrams = corpus2gram

        elif isinstance(corpus1gram, str) and corpus2gram == "":
            self.unigrams = safely_load_json(corpus1gram)
            self.bigrams = safely_load_json(corpus2gram)
        else:
            self.unigrams = None
            self.bigrams = None
            return

        self.N = sum(self.unigrams.values())
        self.L = max_split_length

        self.Pw = Pdist(self.unigrams, self.N, self.unk_probability)
        self.P2w = Pdist(self.bigrams, self.N)

        self.case_split = self.get_compiled()["camel_split"]

    def get_compiled(self):
        try:
            with open(FILE_EXPRESSIONS) as fh:
                expressions = json.load(fh)
        except FileNotFoundError:
            print("File " + str(FILE_EXPRESSIONS) + " not found")

        regexes = {k.lower(): re.compile(expressions[k]) for k, v in expressions.items()}
        return regexes

    def condProbWord(self, word, prev):
        """
        Conditional probability of word, given previous word
        if bigram is not in our list, then fall back to unigrams
        Args:
            word (): candidate word
            prev (): previous observed word
        Returns:
        """
        try:
            return self.P2w[prev + NGRAM_SEP + word] / float(self.Pw[prev])
        except KeyError:
            return self.Pw(word)

    @staticmethod
    def unk_probability(key, total):
        """
        Estimate the probability of an unknown word, penalizing its length
        :param key: the word
        :param total: the count of all tokens
        :return:
        """
        return 10. / (total * 10 ** len(key))

    @staticmethod
    def combine(first, rem):
        """
        Combine first and rem results into one (probability, words) pair
        :param first: a tuple in the form: probability, word
        :param rem: a tuple in the form: probability, list_of_words
        :return:
        """
        (first_prob, first_word) = first
        (rem_prob, rem_words) = rem
        return first_prob + rem_prob, [first_word] + rem_words

    def splits(self, text):
        """
        Return a list of all possible (first, rem) pairs with max length of first <=L
        :param text:
        :return:
        """
        return [(text[:i + 1], text[i + 1:])
                for i in range(min(len(text), self.L))]

    # if you don't have enough RAM lower the maxsize
    @lru_cache(maxsize=65536)
    def find_segment(self, text, prev='<S>'):
        """
        Return (log P(words), words), where words is the best estimated segmentation
        :param text: the text to be segmented
        :param prev:
        :return:
        """
        if not text:
            return 0.0, []
        candidates = [self.combine((log10(self.condProbWord(first, prev)), first), self.find_segment(rem, first))
                      for first, rem in self.splits(text)]
        return max(candidates)

    # if you don't have enough RAM lower the maxsize
    @lru_cache(maxsize=65536)
    def segment(self, word):
        if (self.unigrams is not None and len(self.unigrams) > 0) or (self.bigrams is not None and len(self.bigrams)) > 0:
            if word.islower():
                return " ".join(self.find_segment(word)[1])
            else:
                return self.case_split.sub(r' \1', word).lower()
        else:
            print("Error! Unigrams and/or bigrams not defined")
            return None
