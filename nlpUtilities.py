import json
import re
from functools import reduce
import logging
import os
from string import digits, punctuation

import nltk
import spacy
import subprocess
from collections import OrderedDict
from utils import safely_load_json
from NLPCore.symSpellImplementation import SpellChecker, Verbosity
from NLPCore.textSegmenter import Segmenter
from NLPCore.phraseChecker import PhraseChecker

logger = logging.getLogger(__name__)

ONE_GRAMS_CORPUS = __file__[:__file__.rfind("/")] + "/NLPCorpi/1grams_corpus.json"
if not os.path.exists(ONE_GRAMS_CORPUS):
    ONE_GRAMS_CORPUS = os.path.abspath("") + "/functionalities/sharedFunctionalities/NLPCorpi/1grams_corpus.json"

TWO_GRAMS_CORPUS = __file__[:__file__.rfind("/")] + "/NLPCorpi/2grams_corpus.json"
if not os.path.exists(TWO_GRAMS_CORPUS):
    TWO_GRAMS_CORPUS = os.path.abspath("") + "/functionalities/sharedFunctionalities/NLPCorpi/2grams_corpus.json"

THREE_GRAMS_CORPUS = __file__[:__file__.rfind("/")] + "/NLPCorpi/3grams_corpus.json"
if not os.path.exists(THREE_GRAMS_CORPUS):
    THREE_GRAMS_CORPUS = os.path.abspath("") + "/functionalities/sharedFunctionalities/NLPCorpi/3grams_corpus.json"

SYMSPELL_CORPUS = __file__[:__file__.rfind("/")] + "/NLPCorpi/spell_checker_corpus.txt"
if not os.path.exists(SYMSPELL_CORPUS):
    print(SYMSPELL_CORPUS)
    SYMSPELL_CORPUS = os.path.abspath("") + "/functionalities/sharedFunctionalities/NLPCorpi/spell_checker_corpus.txt"

ABBREVIATIONS_CORPUS = __file__[:__file__.rfind("/")] + "/NLPCorpi/abbreviations.json"
if not os.path.exists(ABBREVIATIONS_CORPUS):
    ABBREVIATIONS_CORPUS = os.path.abspath("") + "/functionalities/sharedFunctionalities/NLPCorpi/abbreviations.json"

CONTRACTIONS_CORPUS = __file__[:__file__.rfind("/")] + "/NLPCorpi/contractions.json"
if not os.path.exists(CONTRACTIONS_CORPUS):
    CONTRACTIONS_CORPUS = os.path.abspath("") + "/functionalities/sharedFunctionalities/NLPCorpi/contractions.json"

VERBS_CORPUS = __file__[:__file__.rfind("/")] + "NLPCorpi/verbs.txt"


class NLPUtilities:
    segmenter = None
    spell_checker = None
    phrase_checker = None
    abbreviations_dictionary = None
    contractions_dictionary = None
    lemmetizer = None
    stemmer = None
    spacy = None
    verbs = None

    @staticmethod
    def initialize_segmenter(monograms: str=ONE_GRAMS_CORPUS, bigrams: str=TWO_GRAMS_CORPUS) -> Segmenter:
        if NLPUtilities.segmenter is None:
            monogram_segmenter_dictionary = safely_load_json(monograms)
            bigram_segmenter_dictionary = safely_load_json(bigrams)
            NLPUtilities.segmenter = Segmenter(monogram_segmenter_dictionary, bigram_segmenter_dictionary)
        return NLPUtilities.segmenter

    @staticmethod
    def initialize_spell_checker(corpus: str=SYMSPELL_CORPUS, max_distance: int=5) -> SpellChecker:
        if NLPUtilities.spell_checker is None:
            NLPUtilities.spell_checker = SpellChecker(corpus, max_dictionary_edit_distance=max_distance)
        return NLPUtilities.spell_checker

    @staticmethod
    def initialize_phrase_checker(corpus: str=THREE_GRAMS_CORPUS) -> PhraseChecker:
        if NLPUtilities.phrase_checker is None:
            NLPUtilities.phrase_checker = PhraseChecker(corpus)
        return NLPUtilities.phrase_checker

    @staticmethod
    def initialize_abbreviations(abbreviations: str=ABBREVIATIONS_CORPUS) -> dict:
        if NLPUtilities.abbreviations_dictionary is None:
            NLPUtilities.abbreviations_dictionary = safely_load_json(abbreviations)
        return NLPUtilities.abbreviations_dictionary

    @staticmethod
    def initialize_contractions(contractions: str=CONTRACTIONS_CORPUS) -> dict:
        if NLPUtilities.contractions_dictionary is None:
            NLPUtilities.contractions_dictionary = safely_load_json(contractions)
        return NLPUtilities.contractions_dictionary

    @staticmethod
    def initialize_verbs(verbs: str=VERBS_CORPUS):
        if NLPUtilities.verbs is None:
            try:
                with open(verbs, "r") as fp:
                    NLPUtilities.verbs = fp.readlines()
            except FileNotFoundError:
                NLPUtilities.verbs = []
        return NLPUtilities.verbs

    @staticmethod
    def initialize_lemmetizer():
        # Download needed corpora, if not already available
        if NLPUtilities.lemmetizer is None:
            nltk.download("punkt")
            nltk.download('averaged_perceptron_tagger')
            nltk.download("wordnet")
            NLPUtilities.lemmetizer = nltk.WordNetLemmatizer()
        return NLPUtilities.lemmetizer

    @staticmethod
    def initialize_stemmer():
        if NLPUtilities.stemmer is None:
            NLPUtilities.stemmer = nltk.PorterStemmer()
        return NLPUtilities.stemmer

    @staticmethod
    def initialize_spacy():
        # Initialize SpaCy
        if NLPUtilities.spacy is None:
            try:
                NLPUtilities.spacy = spacy.load("en_core_web_sm")
            except OSError:
                temp = subprocess.run("python -m spacy download en_core_web_sm",
                                      stderr=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL,
                                      shell=True)
                if temp.stderr != "":
                    subprocess.run("python3.7 -m spacy download en_core_web_sm",
                                   stderr=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL,
                                   shell=True)
                NLPUtilities.spacy = spacy.load("en_core_web_sm")

        return NLPUtilities.spacy

    @staticmethod
    def initialize_all_datasets():
        NLPUtilities.initialize_segmenter()
        NLPUtilities.initialize_spell_checker()
        NLPUtilities.initialize_phrase_checker()
        NLPUtilities.initialize_abbreviations()
        NLPUtilities.initialize_contractions()
        NLPUtilities.initialize_lemmetizer()
        NLPUtilities.initialize_stemmer()
        NLPUtilities.initialize_spacy()
        NLPUtilities.initialize_verbs()

    @staticmethod
    def use_segmenter(phrase: str) -> str:
        segment = NLPUtilities.initialize_segmenter()
        check = NLPUtilities.initialize_spell_checker()
        phrase = phrase.translate(str.maketrans("", "", "-_. ")).lower()
        phrases = [phrase]
        final = ""

        for sub_phrase in phrases:
            try:
                elem = segment.segment(sub_phrase)
                cleanElem = check.word_segmentation(elem, max_edit_distance=2, ignore_token=r"\w+\d")
                if cleanElem is None:
                    return elem.strip()
                original_words = cleanElem.segmented_string.split(" ")
                words = cleanElem.corrected_string.split(" ")

                for index in range(len(words)):
                    if len(words) == len(original_words):
                        final = final + original_words[index] + " "
                    else:
                        if words[index] == original_words[0]:
                            final = final + original_words[0] + " "
                            original_words.pop(0)

                        elif len(original_words) == 0:
                            final = final + words[index]

                        elif (words[index] == original_words[0] + original_words[1] or
                              [index + 1] == original_words[2]) and len(original_words) >= 2:
                            final = final + original_words[0] + original_words[1] + " "
                            original_words.pop(0)
                            original_words.pop(0)

                        else:
                            final = final + original_words[0] + " "
                            original_words.pop(0)

            except Exception as e:
                print("Error " + str(e) + ". subphrase: " + str(sub_phrase))

        return final.strip()

    @staticmethod
    def use_spell_checker(phrase: str) -> OrderedDict:
        sym = NLPUtilities.initialize_spell_checker()
        abbreviations = NLPUtilities.initialize_abbreviations()
        result = OrderedDict()

        for word in phrase.split(" "):
            if word in abbreviations:
                result[word] = {"status": "common abbreviation",
                                "corrected_word": abbreviations[word],
                                "score": 0.7}

            else:
                correctedWord = sym.lookup(word, Verbosity.TOP)
                if len(correctedWord) > 0 and correctedWord[0].term == word:
                    result[word] = {"status": "correct",
                                    "corrected_word": word,
                                    "score": 1}

                elif len(correctedWord) > 0 and word in correctedWord[0].term:
                    result[word] = {"status": "abbreviation",
                                    "corrected_word": correctedWord[0].term,
                                    "score": 0.5}

                else:
                    correctedWord = sym.lookup(word, Verbosity.TOP, max_edit_distance=1)

                    if len(correctedWord) > 0:
                        result[word] = {"status": "mispelled",
                                        "corrected_word": correctedWord[0].term,
                                        "score": 0.3}

                    else:
                        result[word] = {"status": "unknown word",
                                        "corrected_word": "N/A",
                                        "score": 0}
        return result

    @staticmethod
    def use_lemmetizer(phrase: str) -> str:
        """Lemmetize a phrase using nltk. The result is the lemmetized phrase without the punctuation"""
        lemmetizer = NLPUtilities.initialize_lemmetizer()
        # Remove punctuation
        phrase = phrase.translate(str.maketrans("", "", ".!\"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~"))
        # tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(phrase))
        # tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], NLPUtilities.nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                # if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:
                # else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmetizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)

    @staticmethod
    def check_abbreviation(word: str) -> str:
        abbreviations = NLPUtilities.initialize_abbreviations()
        if word in abbreviations:
            return abbreviations[word]
        else:
            return word

    @staticmethod
    def extend_contraction(word: str) -> str:
        contractions = NLPUtilities.initialize_contractions()
        if word in contractions:
            return contractions[word]
        else:
            return word

    @staticmethod
    def use_phrase_checker(phrases: list, phrase_checker=None) -> dict:
        if phrase_checker is None:
            phrase_checker = NLPUtilities.initialize_phrase_checker()

        phrases = [phrase for phrase in phrases if len(phrase) > 0]
        result = {}
        partial_results = []
        score = 0
        count = 0
        for phrase in phrases:
            flag, phrase_result = phrase_checker.check_phrase(phrase)
            if flag is False:
                # logger.error(phrase_result["suggestion"])
                continue
            score = score + sum([elem["score"] for elem in phrase_result])
            count = count + len(phrase_result)
            partial_results.append(phrase_result)
        try:
            score = score / count
        except ZeroDivisionError:
            score = 0
        result["english_phrase_score"] = round(score, 2)
        result["triplets_analysis"] = partial_results
        return result

    @staticmethod
    def use_pos_dependency_tagger(phrases: list) -> list:
        """output: [ [ {word1}, {word2}, .. ], [ phrase2 ], [ phrase3 ], .. ]"""
        tagged_phrases = []
        nlp = NLPUtilities.initialize_spacy()
        if isinstance(phrases, str):
            phrases = [phrases]

        for phrase in phrases:
            analyzed_phrase = nlp(phrase)
            phrase_results = []
            for element in analyzed_phrase:
                phrase_results.append({
                    "word": element.text,
                    "pos_tag": element.pos_,
                    "role": element.dep_,
                    "children": [child for child in element.children]})
            tagged_phrases.append(phrase_results)
        return tagged_phrases

    @staticmethod
    def use_words_stemmer(words: list) -> list:
        stemmer = NLPUtilities.initialize_stemmer()
        result = []
        for word in words:
            result.append({
                "original": word,
                "stemmed": stemmer.stem(word)
            })
        return result

    @staticmethod
    def identifier_checker(words: list) -> list:
        result = []

        for word in words:

            split_word = NLPUtilities.use_segmenter(word)
            score_word = NLPUtilities.use_spell_checker(split_word)
            if 1 < len(split_word.replace(" ", "")) < 25:

                separators_input_vector = NLPUtilities.input_vectorization(split_word, word)

                result_english_score = NLPUtilities.english_score(score_word)
                result_separator_score = NLPUtilities.separator_score(separators_input_vector)

            else:
                if len(word) <= 1:
                    score_word = {word: {"status": "Too short",
                                         "corrected_word": "N/A"}}
                else:
                    score_word = {word: {"status": "Too long",
                                         "corrected_word": "N/A"}}
                result_english_score = 0
                result_separator_score = 0

            # Max possible values are result_english_score = 1 and result_separator_score = 1
            # So final score = (max1 + max2) / 2
            final_score = (result_english_score + result_separator_score) / 2

            if float(final_score) > 0.75:
                final_score = "Very Good"
            elif 0.5 < float(final_score) <= 0.75:
                final_score = "Somewhat Good"
            elif 0.25 < float(final_score) <= 0.5:
                final_score = "Somewhat Bad"
            elif 0 < float(final_score) <= 0.25:
                final_score = "Bad"
            else:
                final_score = "Unknown"

            subwords = []

            for elem in score_word:
                subwords.append({"sub_word": elem,
                                 "status": score_word[elem]["status"],
                                 "suggested_word": score_word[elem]["corrected_word"]})

            word_result = {"name": word,
                           "english_score": str(result_english_score),
                           "separator_score": str(result_separator_score),
                           "sub_words": subwords,
                           "final_score": final_score}

            result.append(word_result)

        return result

    @staticmethod
    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif nltk_tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return None

    @staticmethod
    def input_vectorization(score_word: str, original_word="") -> list:
        # Prepare vectorized words for separators score
        vector = []

        split_word = score_word.split(" ")
        init = 0
        final_phrase = ""
        for index in range(len(original_word)):

            if original_word[init:index].replace("_", "").replace("-", "").lower() == split_word[0]:
                final_phrase = final_phrase + original_word[init:index] + " "
                split_word.pop(0)
                init = index

            if len(split_word) == 1:
                final_phrase = final_phrase + original_word[init:]
                break

        flag = True
        init = True
        for character in final_phrase:
            if flag is True:
                if character == "_":
                    vector.append(5)
                elif character == "-":
                    vector.append(6)
                elif character.isupper():
                    vector.append(7)
                    vector.append(1)
                else:
                    if init is False:
                        vector.append(9)
                    else:
                        init = False
                    vector.append(1)
                flag = False
            else:
                if character == "_":
                    vector.append(2)

                elif character == "-":
                    vector.append(3)

                elif character.isupper():
                    vector.append(4)
                    vector.append(1)

                elif character == " ":
                    flag = True

                else:
                    vector.append(1)

        return vector

    @staticmethod
    def english_score(score: dict) -> float:
        """ Compute total english score """
        final = 0
        for elem in score:
            final = final + score[elem]["score"]

        return final / len(score)

    @staticmethod
    def separator_score(input_vector: list) -> float:
        """ Compute total separators score """
        score = 1
        non_separated_words_penalty = 0.07
        subwords_separators_penalty = 0.2
        other_separators_penalty = 0.15

        # Count number of spaces without separators
        count_spaces = input_vector.count(9)
        score = score - (non_separated_words_penalty * count_spaces)

        # Count number separators within subwords
        count_separators = sum(input_vector.count(sep) for sep in [2, 3, 4])
        score = score - (subwords_separators_penalty * count_separators)

        # Multiple separators types
        words_separators = [7, 6, 5]
        separators = [[sep, input_vector.count(sep)] for sep in words_separators]
        # Most used separator between words
        used_separator = reduce(lambda sep1, sep2: sep1 if sep1[1] > sep2[1] else sep2, separators)
        separators.remove(used_separator)
        other_separators_count = reduce(lambda sep1, sep2: sep1[1] + sep2[1], separators)
        assert isinstance(other_separators_count, int)
        score = score - (other_separators_penalty * other_separators_count)
        score = round(max(min(score, 1), 0), 2)
        return score

    @staticmethod
    def ngrams_maker(source_path: str, n_grams: int=3, corpus_path: str="", lemmetize: bool=False, expand_contraction: bool=False) -> bool:
        nltk.download("wordnet")
        lines = ""
        result = {}

        if os.path.exists(source_path):
            with open(source_path, "r") as fp:
                lines = fp.read()
        else:
            return False

        if not corpus_path:
            corpus_path = str(n_grams) + "grams_corpus"

        if os.path.exists(f"{corpus_path}.json"):
            with open(f"{corpus_path}.json", "r") as fp:
                result = json.load(fp)

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

                if lemmetize:
                    lemm = nltk.WordNetLemmatizer()
                    cleaned = lemm.lemmatize(lemm.lemmatize(lemm.lemmatize(cleaned, pos="n"), pos="v"), pos="a")

                if len(cleaned) == 0:
                    continue

                elif expand_contraction and cleaned in NLPUtilities.contractions:
                    for elem in NLPUtilities.contractions[cleaned].split():
                        temp.append(elem)

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
                freq = nltk.FreqDist(nltk.ngrams(temp, n_grams))
                for elem in freq:
                    te = "_".join(elem)
                    if te in result:
                        result[te] = result[te] + freq[elem]
                    else:
                        result[te] = freq[elem]
            n = n + 1
        #
        # gram = []
        # for phrase in sent:
        #     gram.append(FreqDist(ngrams(phrase, GRAMS_NUMBER)))
        #
        # res = {}
        # for elem in gram:
        #     for p in elem:
        #         te = "_".join(p)
        #         if te in res:
        #             res[te] = res[te] + elem[p]
        #         else:
        #             res[te] = elem[p]
        with open(f"{corpus_path}.json", "w") as fp:
            json.dump(result, fp)
            print(f"N-GRAM CORPUS CREATED: {corpus_path}.json")

if __name__ == "__main__":
    NLPUtilities.ngrams_maker("NLPCorpi/Fellowship_of_the_Ring.txt", n_grams=2)