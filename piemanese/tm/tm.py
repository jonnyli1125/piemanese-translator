import argparse
import os.path
import re
import math
import dill as pickle
import tensorflow as tf
import numpy as np
import Levenshtein

class TranslationModel:
    def __init__(self, replacements=None, en_vocab=None, threshold=0.5,
            tf_model_dir='tm_char_cnn'):
        if not replacements:
            replacements = {}
            replacements_file = f'{os.path.dirname(__file__)}/replacements.tsv'
            with open(replacements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    pi, en = line.strip().split('\t')
                    replacements[pi] = en.split(',')
        self.replacements = replacements
        self.threshold = threshold
        if not en_vocab:
            en_vocab = f'{os.path.dirname(__file__)}/../vocab'
        self.en_vocab = []
        if isinstance(en_vocab, str):
            with open(en_vocab + '/vocab.txt', 'r') as f:
                self.en_vocab = [line.strip() for line in f]
            with open(en_vocab + '/phrase_vocab.txt', 'r') as f:
                self.en_vocab += [line.strip() for line in f]
        else:
            self.en_vocab = en_vocab
        self.tf_model_dir = tf_model_dir
        self.model = tf.keras.models.load_model(tf_model_dir)
        self.word_re = re.compile(r"^([a-z][a-z0-9'&]*)|(\d[a-z0-9'&]*[a-z][a-z0-9'&]*)$")
        self.pi_word_clean_re = re.compile(r'([a-z])\1{2,}')

    def _get_replacements(self, pi_word):
        variations = [
            pi_word,
            re.sub(r'(.)\1+', r'\1', pi_word)
        ]
        found = [w for w in variations if w in self.replacements]
        if not found:
            return None
        replacements = self.replacements[found[0]]
        # TODO precompute replacement probabilities from NN and store in pkl
        return {w: math.log(1/len(replacements)) for w in replacements}

    def logscores(self, pi_word):
        """Compute the TM likelihood p(pi|e) over all e."""
        if pi_word in ['<s>', '</s>'] or not self.word_re.match(pi_word):
            return {pi_word: 0}
        pi_word = self.pi_word_clean_re.sub(r'\1\1', pi_word)
        replacements = self._get_replacements(pi_word)
        if replacements is not None:
            return replacements
        en_vocab = [w for w in self.en_vocab
            if Levenshtein.ratio(pi_word, w)]
        in_tensor = [
            tf.constant([pi_word] * len(en_vocab)),
            tf.constant(en_vocab)
        ]
        probs = self.model.call(in_tensor, training=False).numpy().flatten()
        scores = {en_vocab[i]: p for i, p in enumerate(probs)
            if p >= self.threshold}
        if not scores:
            return {pi_word: 0}
        z = sum(scores.values())
        log_normalized_scores = {w: math.log(p / z) for w, p in scores.items()}
        return log_normalized_scores

    def save(self, path):
        # TODO: save tf model as well
        with open(path, 'wb') as f:
            data = {
                'replacements': self.replacements,
                'threshold': self.threshold,
                'en_vocab': self.en_vocab,
                'tf_model_dir': self.tf_model_dir
            }
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return cls(**pickle.load(f))

if __name__ == '__main__':
    tm = TranslationModel()
    tm.save('tm.pkl')
