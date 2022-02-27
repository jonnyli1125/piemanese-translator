import collections
import csv
import re
import nltk
import dill as pickle

class Translator:
    def __init__(self, replacements_file='replacements.csv', lm_file='lm.pkl'):
        with open(replacements_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            self.replacements = {row[0]: row[1] for row in reader}
        with open(lm_file, 'rb') as f:
            self.lm = pickle.load(f)
        self._init_piemanese_root_forms()

    def __call__(self, text):
        text = self._simple_replace(text)
        text = self._get_best_translation(text)
        return text

    def _simple_replace(self, text):
        """replaces words in text according to replacement dictionary."""
        tokens = text.split()
        new_tokens = []
        for token in tokens:
            token = token.lower()
            word, punc = self._split_punctuation(token)
            word = re.sub(r'^(.{2,})([a-z])\2{2,}$', r'\1\2', word)
            variations = {
                word,
                re.sub(r'([a-z])\1+$', r'\1', word)
            }
            found = variations & self.replacements.keys()
            if found:
                new_word = self.replacements[next(iter(found))]
            else:
                new_word = word
            new_token = new_word + punc
            new_tokens.append(new_token)
        return ' '.join(new_tokens)

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
        word = re.sub(r'er$', 'a', word)
        word = re.sub(r'ing$', 'in', word)
        word = re.sub(r'gh$', '', word)
        word = re.sub(r'ght$', 't', word)
        word = re.sub(r'mb$', 'm', word)
        word = re.sub(r'nt$', 'n', word)
        word = re.sub(r'^wh', 'w', word)
        word = re.sub(r'(?<!^)th', 'f', word)
        word = re.sub(r'([a-z])\1+', r'\1', word)
        word = re.sub(r'[aiueo]', ' ', word)
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
            if word in self.lm.vocab:
                en_tokens.append(pi_tokens[i])
                continue
            candidates = self.pi_root_lookup[self._get_piemanese_root(word)]
            if not candidates:
                en_tokens.append(pi_tokens[i])
                continue
            context = [self._split_punctuation(t)[0] for t in pi_tokens[i-2:i]]
            best_word = max(candidates, key=lambda w: self.lm.score(w, context))
            new_token = best_word + punc
            en_tokens.append(new_token)
        return ' '.join(en_tokens)
