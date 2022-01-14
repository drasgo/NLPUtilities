from nlputilities.utils import safely_load_json
from string import punctuation, digits


class PhraseChecker:
    def __init__(self, three_grams):
        self.three_grams = self._get_corpus(three_grams)
        self.window = []

    @staticmethod
    def _get_corpus(corpus):
        dictionary = safely_load_json(corpus)
        keys = dictionary.keys()
        return [triplet.split("_") for triplet in keys if len(triplet.split("_")) == 3]

    def slide_window(self, words, index):
        if index <= len(words):
            self.window = self.window[1:]
            self.window.append(words[index])
            return True
        else:
            return False

    def phrase_pos_1(self):
        word = self.window[0]
        if any(word == elem[0] for elem in self.three_grams):
            return 1, "N/A"
        else:
            return 0, "N/A"

    def phrase_pos_2(self):
        words = self.window[:2]
        for gram in self.three_grams:
            if all(word in gram[:2] for word in words):
                if words[0] == gram[0] and words[1] == gram[1]:
                    return 1, " ".join(words)
                else:
                    return 0.7, " ".join([words[1], words[0]])

        sub_score, suggestion = self.phrase_pos_1()
        sub_score = max([0, sub_score - 0.6])
        return sub_score, suggestion

    def phrase_pos_3(self):
        for gram in self.three_grams:
            if all(word in gram for word in self.window):
                if self.window[0] == gram[0] and self.window[1] == gram[1] and self.window[2] == gram[2]:
                    score = 1
                elif any(self.window[n] == gram[n] for n in range(len(self.window))):
                    score = 0.7
                else:
                    score = 0.4
                return score, " ".join(gram)

        sub_score, suggestion = self.phrase_pos_2()
        sub_score = max([0, sub_score - 0.6])
        return sub_score, suggestion

    @staticmethod
    def remove_punctuation(phrase):
        return phrase.translate(str.maketrans("", "", digits + punctuation.replace("'", ""))) \
            .lower() \
            .strip()

    @staticmethod
    def append_partial_result(words, score, suggestion):
        return {"words": words,
                "score": score,
                "suggestion": suggestion}

    def check_phrase(self, phrase):
        words = self.remove_punctuation(phrase).split()
        result = []
        # Check 3-grams corpus integrity
        if len(self.three_grams) == 0:
            return False, self.append_partial_result(self.window, 0, "three-grams corpus not available")

        # Check phrase not empty
        if len(words) == 0:
            return False, self.append_partial_result(self.window, 0, "empty phrase")

        # if phrase is less then 3 words, only first word alone and first and second are computer
        elif len(words) <= 2:
            self.window = words

        # else compute triplets of sliding window
        else:
            self.window = words[:3]

        # Compute score and suggestion of first word alone
        first_score, suggestion = self.phrase_pos_1()
        result.append(self.append_partial_result([self.window[0]], first_score, suggestion))

        # Compute score and suggestion of first and second words
        if len(words) > 1:
            second_score, suggestion = self.phrase_pos_1()
            result.append(self.append_partial_result(self.window[:2], second_score, suggestion))

        # Start sliding window process
        for index in range(3, len(words)):
            score, suggestion = self.phrase_pos_3()
            result.append(self.append_partial_result(self.window, score, suggestion))
            if self.slide_window(words, index):
                continue
            else:
                break
        return True, result
