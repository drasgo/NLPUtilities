"""
.. module:: editdistance
   :synopsis: Module for edit distance algorithms.
"""
from enum import Enum
from difflib import SequenceMatcher
from itertools import zip_longest
import re
import numpy as np


class DistanceAlgorithm(Enum):
    """Supported edit distance algorithms"""
    LEVENSHTEIN = 0  #: Levenshtein algorithm.
    DAMERUAUOSA = 1  #: Damerau optimal string alignment algorithm


class EditDistance(object):
    """Edit distance algorithms.

    Parameters
    ----------
    algorithm : :class:`DistanceAlgorithm`
        The distance algorithm to use.

    Attributes
    ----------
    _algorithm : :class:`DistanceAlgorithm`
        The edit distance algorithm to use.
    _distance_comparer : :class:`AbstractDistanceComparer`
        An object to identifier_checker the relative distance between two strings.
        The concrete object will be chosen based on the value of
        :attr:`_algorithm`

    Raises
    ------
    ValueError
        If `algorithm` specifies an invalid distance algorithm.
    """
    def __init__(self, algorithm):
        self._algorithm = algorithm
        if algorithm == DistanceAlgorithm.LEVENSHTEIN:
            self._distance_comparer = Levenshtein()
        elif algorithm == DistanceAlgorithm.DAMERUAUOSA:
            self._distance_comparer = DamerauOsa()
        else:
            raise ValueError("Unknown distance algorithm")

    def compare(self, string_1, string_2, max_distance):
        """Compare a string to the base string to determine the edit
        distance, using the previously selected algorithm.

        Parameters
        ----------
        string_1 : str
            Base string.
        string_2 : str
            The string to compare.
        max_distance : int
            The maximum distance allowed.

        Returns
        -------
        int
            The edit distance (or -1 if `max_distance` exceeded).
        """
        return self._distance_comparer.distance(string_1, string_2,
                                                max_distance)


class AbstractDistanceComparer(object):
    """An interface to identifier_checker relative distance between two strings"""
    def distance(self, string_1, string_2, max_distance):
        """Return a measure of the distance between two strings.

        Parameters
        ----------
        string_1 : str
            One of the strings to compare.
        string_2 : str
            The other string to compare.
        max_distance : int
            The maximum distance that is of interest.

        Returns
        -------
        int
            -1 if the distance is greater than the max_distance, 0 if
            the strings are equivalent, otherwise a positive number
            whose magnitude increases as difference between the strings
            increases.

        Raises
        ------
        NotImplementedError
            If called from abstract class instead of concrete class
        """
        raise NotImplementedError("Should have implemented this")


class Levenshtein(AbstractDistanceComparer):
    """Class providing Levenshtein algorithm for computing edit
    distance metric between two strings

    Attributes
    ----------
    _base_char_1_costs : numpy.ndarray
    """
    def __init__(self):
        self._base_char_1_costs = np.zeros(0, dtype=np.int32)

    def distance(self, string_1, string_2, max_distance):
        """Compute and return the Levenshtein edit distance between two
        strings.

        Parameters
        ----------
        string_1 : str
            One of the strings to compare.
        string_2 : str
            The other string to compare.
        max_distance : int
            The maximum distance that is of interest.

        Returns
        -------
        int
            -1 if the distance is greater than the maxDistance, 0 if
            the strings are equivalent, otherwise a positive number
            whose magnitude increases as difference between the strings
            increases.
        """
        if string_1 is None or string_2 is None:
            return helpers.null_distance_results(string_1, string_2,
                                                 max_distance)
        if max_distance <= 0:
            return 0 if string_1 == string_2 else -1
        max_distance = max_distance = int(min(2 ** 31 - 1, max_distance))
        # if strings of different lengths, ensure shorter string is in
        # string_1. This can result in a little faster speed by
        # spending more time spinning just the inner loop during the
        # main processing.
        if len(string_1) > len(string_2):
            string_2, string_1 = string_1, string_2
        if len(string_2) - len(string_1) > max_distance:
            return -1
        # identify common suffic and/or prefix that can be ignored
        len_1, len_2, start = helpers.prefix_suffix_prep(string_1, string_2)
        if len_1 == 0:
            return len_2 if len_2 <= max_distance else -1

        if len_2 > len(self._base_char_1_costs):
            self._base_char_1_costs = np.zeros(len_2, dtype=np.int32)
        if max_distance < len_2:
            return self._distance_max(string_1, string_2, len_1, len_2,
                                      start, max_distance,
                                      self._base_char_1_costs)
        return self._distance(string_1, string_2, len_1, len_2, start,
                              self._base_char_1_costs)

    def _distance(self, string_1, string_2, len_1, len_2, start,
                  char_1_costs):
        """Internal implementation of the core Levenshtein algorithm.

        **From**: https://github.com/softwx/SoftWx.Match
        """
        char_1_costs = np.asarray([j + 1 for j in range(len_2)])
        current_cost = 0
        for i in range(len_1):
            left_char_cost = above_char_cost = i
            char_1 = string_1[start + i]
            for j in range(len_2):
                # cost of diagonal (substitution)
                current_cost = left_char_cost
                left_char_cost = char_1_costs[j]
                if string_2[start + j] != char_1:
                    # substitution if neither of the two conditions
                    # below
                    if above_char_cost < current_cost:
                        current_cost = above_char_cost
                    if left_char_cost < current_cost:
                        current_cost = left_char_cost
                    current_cost += 1
                char_1_costs[j] = above_char_cost = current_cost
        return current_cost

    def _distance_max(self, string_1, string_2, len_1, len_2, start,
                      max_distance, char_1_costs):
        """Internal implementation of the core Levenshtein algorithm
        that accepts a max_distance.

        **From**: https://github.com/softwx/SoftWx.Match
        """
        char_1_costs = np.asarray([j + 1 if j < max_distance
                                   else max_distance + 1
                                   for j in range(len_2)])
        len_diff = len_2 - len_1
        j_start_offset = max_distance - len_diff
        j_start = 0
        j_end = max_distance
        current_cost = 0
        for i in range(len_1):
            char_1 = string_1[start + i]
            prev_char_1_cost = above_char_cost = i
            # no need to look beyond window of lower right
            # diagonal - max_distance cells (lower right diag is
            # i - lenDiff) and the upper left diagonal +
            # max_distance cells (upper left is i)
            j_start += 1 if i > j_start_offset else 0
            j_end += 1 if j_end < len_2 else 0
            for j in range(j_start, j_end):
                # cost of diagonal (substitution)
                current_cost = prev_char_1_cost
                prev_char_1_cost = char_1_costs[j]
                if string_2[start + j] != char_1:
                    # substitution if neither of the two conditions
                    # below
                    if above_char_cost < current_cost:
                        current_cost = above_char_cost
                    if prev_char_1_cost < current_cost:
                        current_cost = prev_char_1_cost
                    current_cost += 1
                char_1_costs[j] = above_char_cost = current_cost
            if char_1_costs[i + len_diff] > max_distance:
                return -1
        return current_cost if current_cost <= max_distance else -1


class DamerauOsa(AbstractDistanceComparer):
    """Class providing optimized methods for computing
    Damerau-Levenshtein Optimal String Alignment (OSA) comparisons
    between two strings.

    Attributes
    ----------
    _base_char_1_costs : numpy.ndarray
    _base_prev_char_1_costs : numpy.ndarray

    """
    def __init__(self):
        self._base_char_1_costs = np.zeros(0, dtype=np.int32)
        self._base_prev_char_1_costs = np.zeros(0, dtype=np.int32)

    def distance(self, string_1, string_2, max_distance):
        """Compute and return the Damerau-Levenshtein optimal string
        alignment edit distance between two strings.

        Parameters
        ----------
        string_1 : str
            One of the strings to compare.
        string_2 : str
            The other string to compare.
        max_distance : int
            The maximum distance that is of interest.

        Returns
        -------
        int
            -1 if the distance is greater than the maxDistance, 0 if
            the strings are equivalent, otherwise a positive number
            whose magnitude increases as difference between the strings
            increases.
        """
        if string_1 is None or string_2 is None:
            return helpers.null_distance_results(string_1, string_2,
                                                 max_distance)
        if max_distance <= 0:
            return 0 if string_1 == string_2 else -1
        max_distance = int(min(2 ** 31 - 1, max_distance))
        # if strings of different lengths, ensure shorter string is in
        # string_1. This can result in a little faster speed by
        # spending more time spinning just the inner loop during the
        # main processing.
        if len(string_1) > len(string_2):
            string_2, string_1 = string_1, string_2
        if len(string_2) - len(string_1) > max_distance:
            return -1
        # identify common suffix and/or prefix that can be ignored
        len_1, len_2, start = helpers.prefix_suffix_prep(string_1, string_2)
        if len_1 == 0:
            return len_2 if len_2 <= max_distance else -1

        if len_2 > len(self._base_char_1_costs):
            self._base_char_1_costs = np.zeros(len_2, dtype=np.int32)
            self._base_prev_char_1_costs = np.zeros(len_2, dtype=np.int32)
        if max_distance < len_2:
            return self._distance_max(string_1, string_2, len_1, len_2,
                                      start, max_distance,
                                      self._base_char_1_costs,
                                      self._base_prev_char_1_costs)
        return self._distance(string_1, string_2, len_1, len_2, start,
                              self._base_char_1_costs,
                              self._base_prev_char_1_costs)

    def _distance(self, string_1, string_2, len_1, len_2, start,
                  char_1_costs, prev_char_1_costs):
        """Internal implementation of the core Damerau-Levenshtein,
        optimal string alignment algorithm.

        **From**: https://github.com/softwx/SoftWx.Match
        """
        char_1_costs = np.asarray([j + 1 for j in range(len_2)])
        char_1 = " "
        current_cost = 0
        for i in range(len_1):
            prev_char_1 = char_1
            char_1 = string_1[start + i]
            char_2 = " "
            left_char_cost = above_char_cost = i
            next_trans_cost = 0
            for j in range(len_2):
                this_trans_cost = next_trans_cost
                next_trans_cost = prev_char_1_costs[j]
                # cost of diagonal (substitution)
                prev_char_1_costs[j] = current_cost = left_char_cost
                # left now equals current cost (which will be diagonal
                # at next iteration)
                left_char_cost = char_1_costs[j]
                prev_char_2 = char_2
                char_2 = string_2[start + j]
                if char_1 != char_2:
                    # substitution if neither of two conditions below
                    if above_char_cost < current_cost:
                        current_cost = above_char_cost
                    if left_char_cost < current_cost:
                        current_cost = left_char_cost
                    current_cost += 1
                    if (i != 0 and j != 0
                            and char_1 == prev_char_2
                            and prev_char_1 == char_2
                            and this_trans_cost + 1 < current_cost):
                        # transposition
                        current_cost = this_trans_cost + 1
                char_1_costs[j] = above_char_cost = current_cost
        return current_cost

    def _distance_max(self, string_1, string_2, len_1, len_2, start,
                      max_distance, char_1_costs, prev_char_1_costs):
        """Internal implementation of the core Damerau-Levenshtein,
        optimal string alignment algorithm that accepts a max_distance.

        **From**: https://github.com/softwx/SoftWx.Match
        """
        char_1_costs = np.asarray([j + 1 if j < max_distance
                                   else max_distance + 1
                                   for j in range(len_2)])
        len_diff = len_2 - len_1
        j_start_offset = max_distance - len_diff
        j_start = 0
        j_end = max_distance
        char_1 = " "
        current_cost = 0
        for i in range(len_1):
            prev_char_1 = char_1
            char_1 = string_1[start + i]
            char_2 = " "
            left_char_cost = above_char_cost = i
            next_trans_cost = 0
            # no need to look beyond window of lower right diagonal -
            # max_distance cells (lower right diag is i - len_diff) and
            # the upper left diagonal + max_distance cells (upper left
            # is i)
            j_start += 1 if i > j_start_offset else 0
            j_end += 1 if j_end < len_2 else 0
            for j in range(j_start, j_end):
                this_trans_cost = next_trans_cost
                next_trans_cost = prev_char_1_costs[j]
                # cost of diagonal (substitution)
                prev_char_1_costs[j] = current_cost = left_char_cost
                # left now equals current cost (which will be diagonal
                # at next iteration)
                left_char_cost = char_1_costs[j]
                prev_char_2 = char_2
                char_2 = string_2[start + j]
                if char_1 != char_2:
                    # substitution if neither of two conditions below
                    if above_char_cost < current_cost:
                        current_cost = above_char_cost
                    if left_char_cost < current_cost:
                        current_cost = left_char_cost
                    current_cost += 1
                    if (i != 0 and j != 0 and char_1 == prev_char_2
                            and prev_char_1 == char_2
                            and this_trans_cost + 1 < current_cost):
                        # transposition
                        current_cost = this_trans_cost + 1
                char_1_costs[j] = above_char_cost = current_cost
            if char_1_costs[i + len_diff] > max_distance:
                return -1
        return current_cost if current_cost <= max_distance else -1


"""
.. module:: helpers
   :synopsis: Helper functions
"""


class helpers:
    @staticmethod
    def null_distance_results(string1, string2, max_distance):
        """Determines the proper return value of an edit distance function
        when one or both strings are null.

        Parameters
        ----------
        string1 : str
            Base string.
        string2 : str
            The string to compare.
        max_distance : int
            The maximum distance allowed.

        Returns
        -------
        int
            -1 if the distance is greater than the max_distance, 0 if the
            strings are equivalent (both are None), otherwise a positive
            number whose magnitude is the length of the string which is not
            None.
        """
        if string1 is None:
            if string2 is None:
                return 0
            else:
                return len(string2) if len(string2) <= max_distance else -1
        return len(string1) if len(string1) <= max_distance else -1

    @staticmethod
    def prefix_suffix_prep(string1, string2):
        """Calculates starting position and lengths of two strings such
        that common prefix and suffix substrings are excluded.
        Expects len(string1) <= len(string2)

        Parameters
        ----------
        string1 : str
            Base string.
        string2 : str
            The string to compare.

        Returns
        -------
        len1, len2, start : (int, int, int)
            `len1` and len2` are lengths of the part excluding common
            prefix and suffix, and `start` is the starting position.
        """
        # this is also the minimun length of the two strings
        len1 = len(string1)
        len2 = len(string2)
        # suffix common to both strings can be ignored
        while len1 != 0 and string1[len1 - 1] == string2[len2 - 1]:
            len1 -= 1
            len2 -= 1
        # prefix common to both strings can be ignored
        start = 0
        while start != len1 and string1[start] == string2[start]:
            start += 1
        if start != 0:
            len1 -= start
            # length of the part excluding common prefix and suffix
            len2 -= start
        return len1, len2, start

    @staticmethod
    def to_similarity(distance, length):
        """Calculate a similarity measure from an edit distance.

        Parameters
        ----------
        distance : int
            The edit distance between two strings.
        length : int
            The length of the longer of the two strings the edit distance
            is from.

        Returns
        -------
        float
            A similarity value from 0 to 1.0 (1 - (length / distance)),
            -1 if distance is negative
        """
        return -1 if distance < 0 else 1.0 - distance / length

    @staticmethod
    def try_parse_int64(string):
        """Converts the string representation of a number to its 64-bit
        signed integer equivalent.

        Parameters
        ----------
        string : str
            string representation of a number

        Returns
        -------
        int or None
            The 64-bit signed integer equivalent, or None if conversion
            failed or if the number is less than the min value or greater
            than the max value of a 64-bit signed integer.
        """
        try:
            ret = int(string)
        except ValueError:
            return None
        return None if ret < -2 ** 64 or ret >= 2 ** 64 else ret

    @staticmethod
    def parse_words(phrase, preserve_case=False):
        """Create a non-unique wordlist from sample text. Language
        independent (e.g. works with Chinese characters)

        Parameters
        ----------
        phrase : str
            Sample text that could contain one or more words
        preserve_case : bool, optional
            A flag to determine if we can to preserve the cases or convert
            all to lowercase

        Returns
        -------
        list
            A list of words
        """
        # \W non-words, use negated set to ignore non-words and "_"
        # (underscore). Compatible with non-latin characters, does not
        # split words at apostrophes
        if preserve_case:
            return re.findall(r"([^\W_]+['’]*[^\W_]*)", phrase)
        else:
            return re.findall(r"([^\W_]+['’]*[^\W_]*)", phrase.lower())

    @staticmethod
    def is_acronym(word):
        """Checks is the word is all caps (acronym) and/or contain numbers

        Parameters
        ----------
        word : str
            The word to check

        Returns
        -------
        bool
            True if the word is all caps and/or contain numbers, e.g.,
            ABCDE, AB12C. False if the word contains lower case letters,
            e.g., abcde, ABCde, abcDE, abCDe, abc12, ab12c
        """
        return re.match(r"\b[A-Z0-9]{2,}\b", word) is not None

    @staticmethod
    def transfer_casing_for_matching_text(text_w_casing, text_wo_casing):
        """Transferring the casing from one text to another - assuming that
        they are 'matching' texts, alias they have the same length.

        Parameters
        ----------
        text_w_casing : str
            Text with varied casing
        text_wo_casing : str
            Text that is in lowercase only

        Returns
        -------
        str
            Text with the content of `text_wo_casing` and the casing of
            `text_w_casing`

        Raises
        ------
        ValueError
            If the input texts have different lengths
        """
        if len(text_w_casing) != len(text_wo_casing):
            raise ValueError("The 'text_w_casing' and 'text_wo_casing' "
                             "don't have the same length, "
                             "so you can't use them with this method, "
                             "you should be using the more general "
                             "transfer_casing_similar_text() method.")

        return ''.join([y.upper() if x.isupper() else y.lower()
                        for x, y in zip(text_w_casing, text_wo_casing)])

    @staticmethod
    def transfer_casing_for_similar_text(text_w_casing, text_wo_casing):
        """Transferring the casing from one text to another - for similar
        (not matching) text

        1. It will use `difflib`'s `SequenceMatcher` to identify the
           different type of changes needed to turn `text_w_casing` into
           `text_wo_casing`
        2. For each type of change:

           - for inserted sections:

             - it will transfer the casing from the prior character
             - if no character before or the character before is the\
               space, then it will transfer the casing from the following\
               character

           - for deleted sections: no case transfer is required
           - for equal sections: just swap out the text with the original,\
             the one with the casings, as otherwise the two are the same
           - replaced sections: transfer the casing using\
             :meth:`transfer_casing_for_matching_text` if the two has the\
             same length, otherwise transfer character-by-character and\
             carry the last casing over to any additional characters.

        Parameters
        ----------
        text_w_casing : str
            Text with varied casing
        text_wo_casing : str
            Text that is in lowercase only

        Returns
        -------
        text_wo_casing : str
            If `text_wo_casing` is empty
        c : str
            Text with the content of `text_wo_casing` but the casing of
            `text_w_casing`

        Raises
        ------
        ValueError
            If `text_w_casing` is empty
        """
        if not text_wo_casing:
            return text_wo_casing

        if not text_w_casing:
            raise ValueError("We need 'text_w_casing' to know what "
                             "casing to transfer!")

        _sm = SequenceMatcher(None, text_w_casing.lower(),
                              text_wo_casing)

        # we will collect the case_text:
        c = ''

        # get the operation codes describing the differences between the
        # two strings and handle them based on the per operation code rules
        for tag, i1, i2, j1, j2 in _sm.get_opcodes():
            # Print the operation codes from the SequenceMatcher:
            # print('{:7}   a[{}:{}] --> b[{}:{}] {!r:>8} --> {!r}'
            #       .format(tag, i1, i2, j1, j2,
            #               text_w_casing[i1:i2],
            #               text_wo_casing[j1:j2]))

            # inserted character(s)
            if tag == 'insert':
                # if this is the first character and so there is no
                # character on the left of this or the left of it a space
                # then take the casing from the following character
                if i1 == 0 or text_w_casing[i1 - 1] == ' ':
                    if text_w_casing[i1] and text_w_casing[i1].isupper():
                        c += text_wo_casing[j1:j2].upper()
                    else:
                        c += text_wo_casing[j1:j2].lower()
                else:
                    # otherwise just take the casing from the prior
                    # character
                    if text_w_casing[i1 - 1].isupper():
                        c += text_wo_casing[j1:j2].upper()
                    else:
                        c += text_wo_casing[j1:j2].lower()

            elif tag == 'delete':
                # for deleted characters we don't need to do anything
                pass

            elif tag == 'equal':
                # for 'equal' we just transfer the text from the
                # text_w_casing, as anyhow they are equal (without the
                # casing)
                c += text_w_casing[i1:i2]

            elif tag == 'replace':
                _w_casing = text_w_casing[i1:i2]
                _wo_casing = text_wo_casing[j1:j2]

                # if they are the same length, the transfer is easy
                if len(_w_casing) == len(_wo_casing):
                    c += helpers.transfer_casing_for_matching_text(
                        text_w_casing=_w_casing, text_wo_casing=_wo_casing)
                else:
                    # if the replaced has a different length, then we
                    # transfer the casing character-by-character and using
                    # the last casing to continue if we run out of the
                    # sequence
                    _last = 'lower'
                    for w, wo in zip_longest(_w_casing, _wo_casing):
                        if w and wo:
                            if w.isupper():
                                c += wo.upper()
                                _last = 'upper'
                            else:
                                c += wo.lower()
                                _last = 'lower'
                        elif not w and wo:
                            # once we ran out of 'w', we will carry over
                            # the last casing to any additional 'wo'
                            # characters
                            c += wo.upper() if _last == 'upper' else wo.lower()
        return c
