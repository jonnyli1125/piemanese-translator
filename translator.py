import csv
import re
import nltk
import dill as pickle
import phonemizer
import panphon.distance

class Translator:
    def __init__(self, replacements_file='replacements.csv', lm_file='lm.pkl',
        pron_file='pron.pkl'):
        with open(replacements_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            self.replacements = {row[0]: row[1].split(',') for row in reader}
        with open(lm_file, 'rb') as f:
            self.lm = pickle.load(f)
        with open(pron_file, 'rb') as f:
            self.pron_lookup = pickle.load(f)
        self.dst = panphon.distance.Distance()
        self.dst.fm.weights[2] = 2.0  # consonant
        self.dst.fm.weights[6] = 10.0  # nasal
        self.dst.fm.weights[14] = 10.0  # labial

    def __call__(self, text):
        return self._get_best_translation(text)

    def _phonemize(self, text):
        text = phonemizer.phonemize(text).strip()
        text = text.replace('ɚ', 'ə˞').replace('ɝ', 'ɜ˞')
        return text

    def _get_penalty(self, pron1, pron2, word1, word2, alpha=.05, beta=5):
        """
        return pronunciation and grapheme based penalty between two words.
        """
        pi1 = self._get_piemanese_root(word1)
        pi2 = self._get_piemanese_root(word2)
        return (alpha * self.dst.weighted_feature_edit_distance(pron1, pron2) +
            beta * self.dst.fast_levenshtein_distance(pi1, pi2))

    def _get_tm_scores(self, word, max_d=3):
        """
        compute the translation model likelihoods p(pi|e) over all e.

        since it is expensive to compute over all words in e, we only take
        words that have pronunciation levenshtein distance within max_d of the
        pronunciation of word.

        scores (or penalties) are computed by self._get_penalty.
        """
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
            return {w: 0 for w in self.replacements[found[0]]}
        elif word in self.lm.vocab or not re.match(r'^[a-z]+$', word):
            return {}
        else:
            pron = self._phonemize(word)
            guesses = [p for p in self.pron_lookup
                if self.dst.fast_levenshtein_distance(pron, p) <= max_d]
            candidates = {}
            for p in guesses:
                for w in self.pron_lookup[p]:
                    candidates[w] = -self._get_penalty(pron, p, word, w)
            return candidates

    def _split_punctuation(self, word):
        """splits a token into word, punctuation"""
        match = re.match(r'^(\w+)([?.!]*)$', word)
        if match:
            return match.group(1), match.group(2)
        else:
            return word, ''

    def _get_piemanese_root(self, word):
        """
        get piemanese root form of word (consonant grapheme sequence).
        used for grapheme based diff in penalty calculation.
        """
        V = 'aiueo'
        C = 'bcdfghjklmnpqrstvwxz'
        word = re.sub(r'er(ed)$', 'a\1', word)
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
        word = re.sub(rf'([{V}][{C}]{1,2})(e|u)(s?)$', r'\1\3', word)
        #word = re.sub(rf'[{vowels}]|((?<!^)y)', '', word)
        return word

    def _get_best_translation(self, pi_words, verbose=True, k=10):
        """
        returns best english translation by greedily decoding.
        in mathematical terms this is:
            argmax_e p(pi|e) * p(e)
            = argmax_e p(e|pi)

        if verbose is true, show top k words in log.
        """
        pi_tokens = ['<s>', '<s>'] + pi_words.split()
        en_tokens = []
        for i in range(2, len(pi_tokens)):
            word, punc = self._split_punctuation(pi_tokens[i])
            tm_scores = self._get_tm_scores(word)
            if not tm_scores:
                en_tokens.append(pi_tokens[i])
                continue
            context = [self._split_punctuation(t)[0] for t in en_tokens[-2:]]
            combined_scores = {w: self.lm.logscore(w, context) + tm_scores[w]
                for w in tm_scores}
            if verbose:
                print(sorted(combined_scores.items(), key=lambda x: -x[1])[:k])
            best_word = max(combined_scores, key=combined_scores.get)
            new_token = best_word + punc
            en_tokens.append(new_token)
        return ' '.join(en_tokens)
