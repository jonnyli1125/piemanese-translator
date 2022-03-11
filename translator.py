import collections
import csv
import re
import nltk
import dill as pickle

class Translator:
    def __init__(self, replacements_file='replacements.csv', lm_file='lm.pkl'):
        with open(replacements_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            self.replacements = {row[0]: row[1].split(',') for row in reader}
        with open(lm_file, 'rb') as f:
            self.lm = pickle.load(f)
        self._init_piemanese_root_forms()

    def __call__(self, text):
        return self._get_best_translation(text)

    def _lookup_candidates(self, word):
        word = re.sub(r'^(.{2,})([a-z])\2{2,}$', r'\1\2', word)
        variations = [
            word,
            re.sub(r'([a-z])\1+', r'\1', word),
            word.replace('i', 'ee'),
            word.replace('ee', 'i'),
            word.replace('oo', 'u'),
            word.replace('u', 'oo')
        ]
        found = [w for w in variations if w in self.replacements]
        if found:
            return self.replacements[found[0]]
        else:
            return self.pi_root_lookup[self._get_piemanese_root(word)]

    def _split_punctuation(self, word):
        """splits a token into word, punctuation"""
        match = re.match(r'^(\w+)([?.!]*)$', word)
        if match:
            return match.group(1), match.group(2)
        else:
            return word, ''

    def _init_piemanese_root_forms(self):
        """
        precompute the translation model likelihoods p(pi|e) for all e.
        in this case for simplicity's sake, p(pi|e) is either

        - 1, if pi is a valid piemanese translation of e, or
        - 0, otherwise

        pi is a valid piemanese translation of e iff the piemanese root
        form of pi is equal to the piemanese root form of e.

        therefore, this is equivalent to computing the set:
            { e in vocab | pi_root(pi) = pi_root(e) }, for some pi

        however, since it is costly to calculate pi_root(e) for all e,
        we can precompute it and store it as a reverse lookup dictionary.
        """
        self.pi_root_lookup = collections.defaultdict(list)
        for word in self.lm.vocab:
            pi_word = self._get_piemanese_root(word)
            self.pi_root_lookup[pi_word].append(word)

    def _get_piemanese_root(self, word):
        """get piemanese root form of word."""
        vowels = 'aiueo'
        consonants = 'bcdfghjklmnpqrstvwxz'
        word = re.sub(r'er$', 'a', word)
        word = re.sub(r'ing$', 'in', word)
        word = re.sub(r'ph', 'f', word)
        word = re.sub(r'gh$', '', word)
        word = re.sub(r'ght$', 't', word)
        word = re.sub(r'mb$', 'm', word)
        word = re.sub(r'nt$', 'n', word)
        word = re.sub(r'ck$', 'c', word)
        word = re.sub(r'^wh', 'w', word)
        word = re.sub(r'(?<!^)th', 'f', word)
        word = re.sub(r'([a-z])\1+', r'\1', word)
        word = re.sub(rf'([{vowels}][{consonants}]{1,2})e(s?)$', r'\1\2', word)
        word = re.sub(rf'[{vowels}]|((?<!^)y)', ' ', word)
        return re.sub(r'\s+', ' ', word)

    def _get_best_translation(self, pi_words):
        """
        returns best english translation by greedily decoding.
        in mathematical terms this is:
            argmax_e p(pi|e) * p(e)
            = argmax_e p(e|pi)
        """
        pi_tokens = ['<s>', '<s>'] + pi_words.split()
        en_tokens = []
        for i in range(2, len(pi_tokens)):
            word, punc = self._split_punctuation(pi_tokens[i])
            if word in self.lm.vocab and word not in self.replacements:
                en_tokens.append(pi_tokens[i])
                continue
            candidates = self._lookup_candidates(word)
            if not candidates:
                en_tokens.append(pi_tokens[i])
                continue
            context = [self._split_punctuation(t)[0] for t in pi_tokens[i-2:i]]
            best_word = max(candidates, key=lambda w: self.lm.score(w, context))
            new_token = best_word + punc
            en_tokens.append(new_token)
        return ' '.join(en_tokens)
