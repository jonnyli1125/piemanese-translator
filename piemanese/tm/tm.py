import argparse
import os.path
import re
import math
import functools
import dill as pickle
import tensorflow as tf

class TranslationModel:
    def __init__(self, replacements=None, en_vocab=None,
            tf_model_dir='tm_lstm'):
        if not replacements:
            replacements = {}
            replacements_file = f'{os.path.dirname(__file__)}/replacements.tsv'
            with open(replacements_file, 'r', encoding='utf-8') as f:
                for line in f:
                    pi, en = line.strip().split('\t')
                    replacements[pi] = en.split(',')
        self.replacements = replacements
        self.en_vocab = []
        vocab_dir = f'{os.path.dirname(__file__)}/../vocab'
        if not en_vocab:
            en_vocab = vocab_dir
        if isinstance(en_vocab, str):
            with open(f'{en_vocab}/vocab.txt', 'r') as f:
                self.en_vocab = [line.strip() for line in f]
            with open(f'{en_vocab}/phrase_vocab.txt', 'r') as f:
                self.en_vocab += [line.strip() for line in f]
            with open(f'{en_vocab}/names.txt', 'r') as f:
                self.en_vocab += [line.strip() for line in f]
            self.en_vocab = list(dict.fromkeys(self.en_vocab))
        else:
            self.en_vocab = en_vocab
        self.tf_model_dir = tf_model_dir
        self.model = tf.keras.models.load_model(tf_model_dir)
        self.word_re = re.compile(r"^[a-z][a-z0-9']*$")
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
        return {w: 1 / len(replacements) for w in replacements}

    def clean_words(self, pi_words):
        return [self.pi_word_clean_re.sub(r'\1\1', w) for w in pi_words]

    def multiple_scores(self, pi_words, threshold=0.5):
        """Compute the TM likelihood p(pi|e) over all e, for all inputs pi."""
        scores = {}
        # get vocab lengths so we can split output tensor afterwards
        en_vocabs = [self.en_vocab]
        en_vocab_lengths = []
        en_vocab_all = []
        pi_words_all = []
        for pi_word in pi_words:
            if pi_word in scores:
                continue
            if pi_word in ['<s>', '</s>']:
                scores[pi_word] = {pi_word: 1}
                continue
            replacements = self._get_replacements(pi_word)
            if replacements is not None:
                scores[pi_word] = replacements
                continue
            if not self.word_re.match(pi_word):
                scores[pi_word] = {pi_word: 1}
                continue
            pi_chars = set(pi_word)
            for en_vocab in en_vocabs:
                # filter by heuristic first: levenshtein ratio > 0
                # e.g. there is at least one common character
                en_heur = [w for w in en_vocab if any(c in pi_chars for c in w)]
                en_vocab_all += en_heur
                en_vocab_lengths.append(len(en_heur))
                pi_words_all += [pi_word] * len(en_heur)
        # tf model call
        if not pi_words_all:
            return scores
        in_tensor = [
            tf.constant(pi_words_all),
            tf.constant(en_vocab_all)
        ]
        out_tensor = self.model.call(in_tensor, training=False)
        out_probs = tf.reshape(out_tensor, [-1]).numpy()
        # split by vocab lengths
        i = 0
        for n in en_vocab_lengths:
            if n == 0:
                continue
            pi_word = pi_words_all[i]
            if pi_word not in scores:
                en_scores = {en_vocab_all[j]: out_probs[j]
                    for j in range(i, i+n) if out_probs[j] >= threshold}
                if not en_scores:
                    en_scores = {pi_word: 1}
                scores[pi_word] = en_scores
            i += n
        return scores

    def save(self, path):
        # TODO: save tf model as well
        with open(path, 'wb') as f:
            data = {
                'replacements': self.replacements,
                'en_vocab': self.en_vocab,
                'tf_model_dir': self.tf_model_dir
            }
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return cls(**pickle.load(f))

def debug(args):
    tm = TranslationModel.load(args.tm_file)
    while True:
        words = input().strip().lower().split()
        scores = tm.multiple_scores(words)
        for pi_word, en_word_scores in scores.items():
            print(pi_word, sorted(en_word_scores.items(), key=lambda x: -x[1]))

def train(args):
    tm = TranslationModel()
    tm.save(args.tm_file)
    print('Saved to', args.tm_file)
