from nltk import FreqDist, ngrams, download
import os
import json
import re
from string import punctuation, digits

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have",
    "he's": "he has",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has",
    "I'd": "I had",
    "I'd've": "I would have",
    "I'll": "I shall",
    "I'll've": "I shall have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}

# Change this to specify your text, which is going to be the source of the n-grams corpus
# Default book is Fellowship of the Ring, by Tolkienn
SOURCE_TEXT_PATH = "/nlprocessors/functionalities/sharedFunctionalities/NLPCorpi/corpusMakers/test3.txt"
# Specify the number of grams there will be. Default is 3
GRAMS_NUMBER = 3
# Specify corpus name. Default is (grams_number)_corpus.json
CORPUS_NAME = str(GRAMS_NUMBER) + "grams_corpus.json"


def make_corpus(source_file: str=SOURCE_TEXT_PATH, grams_number: int=GRAMS_NUMBER, corpus_name: str=CORPUS_NAME):
    download("wordnet")
    lines = ""
    result = {}

    if os.path.exists(SOURCE_TEXT_PATH):
        with open(SOURCE_TEXT_PATH, "r") as fp:
            lines = fp.read()
    else:
        print("Source file doesn't exist: " + SOURCE_TEXT_PATH + ". Exiting")
        exit()

    if os.path.exists(CORPUS_NAME):
        with open(CORPUS_NAME, "r") as fp:
            result = json.load(fp)
            print("Loaded previous " + str(GRAMS_NUMBER) + " corpus: " + CORPUS_NAME)

    n = 0
    for phrase in lines.split("."):
        if n % 10000 == 0:
            print("" + str(n) + " phrases computed")
        temp = []
        for word in phrase.split():

            cleaned = word.translate(str.maketrans("", "", digits + punctuation.replace("'", "") + "♞♟")) \
                .lower() \
                .strip()

            regrex_pattern = re.compile(pattern="["
                                                u"\U0001F600-\U0001F64F"  # emoticons
                                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                                "]+", flags=re.UNICODE)
            cleaned = regrex_pattern.sub(r'', cleaned)

            # Uncomment this if you want your corpus to be completely lemmatized. E.g. :
            # are/am/is --> be
            # mice --> mouse
            # people --> person
            # better --> good
            # cleaned = lemm.lemmatize(lemm.lemmatize(lemm.lemmatize(cleaned, pos="n"), pos="v"), pos="a")

            if len(cleaned) == 0:
                continue

            # Uncomment this if you want to expand every contraction. E.g.
            # doesn't --> does not
            # elif cleaned in contractions:
            #     for elem in contractions[cleaned].split():
            #         temp.append(elem)

            else:
                if cleaned[0] == "'":
                    cleaned = cleaned[1:]
                try:
                    if cleaned[-1] == "'":
                        cleaned = cleaned[:-1]
                except IndexError:
                    continue
                temp.append(cleaned)
        if len(temp) > 0:
            freq = FreqDist(ngrams(temp, GRAMS_NUMBER))
            for elem in freq:
                te = "_".join(elem)
                if te in result:
                    result[te] = result[te] + freq[elem]
                else:
                    result[te] = freq[elem]
            # sent.append(temp)
        n = n + 1
    with open(CORPUS_NAME, "w") as fp:
        json.dump(result, fp)
        print("N-GRAM CORPUS CREATED: " + CORPUS_NAME)


if __name__ == "__main__":
    make_corpus()
