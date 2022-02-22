import csv
import re
import nltk
import dill as pickle

class Translator:
    def __init__(self, replacements_file='replacements.csv', lm_file='lm.pkl'):
        """
        initialize replacements dictionary, initialize trigram LM,
        generate piemanese perturbations on vocab.
        """
        with open(replacements_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            self.replacements = {row[0]: row[1] for row in reader}
        with open(lm_file, 'rb') as f:
            self.lm = pickle.load(f)

    def __call__(self, text):
        text = self._simple_replace(text)
        text = self._get_best_translation(text)
        return text

    def _simple_replace(self, text):
        """
        replaces words in text according to replacement dictionary.
        """
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
        """
        splits a token into word, punctuation
        """
        match = re.match(r'^(\w+)([?.!]*)$', word)
        if match:
            return match.group(1), match.group(2)
        else:
            return word, ''

    def _is_valid_translation(self, pi_word, en_word):
        """
        returns the translation model likelihood p(pi|e), which in this case
        for simplicity's sake is either

        - 1, if pi is a valid piemanese perturbation of e, or
        - 0, otherwise

        pi is a valid piemanese perturbation of e iff the standardized piemanese
        form of pi is equal to the standardized piemanese form of e.
        """
        return self._to_piemanese(pi_word) == self._to_piemanese(en_word)

    def _to_piemanese(self, word):
        """
        get standardized piemanese form of word.
        """
        word = re.sub(r'er$', 'a', word)
        word = re.sub(r'ing$', 'in', word)
        word = re.sub(r'gh$', '', word)
        word = re.sub(r'ght$', 't', word)
        word = re.sub(r'mb$', 'm', word)
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
            candidates = [w for w in self.lm.vocab
                          if self._is_valid_translation(word, w)]
            if not candidates:
                en_tokens.append(pi_tokens[i])
                continue
            context = [self._split_punctuation(t)[0] for t in pi_tokens[i-2:i]]
            best_word = max(candidates, key=lambda w: self.lm.score(w, context))
            new_token = best_word + punc
            en_tokens.append(new_token)
        return ' '.join(en_tokens)
