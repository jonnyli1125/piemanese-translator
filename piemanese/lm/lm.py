import argparse
import math
import glob
import fileinput
from collections import Counter
from nltk.lm.preprocessing import padded_everygram_pipeline
import dill as pickle

class NgramCounts:
    """Object to store ngram counts."""
    def __init__(self):
        self.counts = Counter()
        self.counts_of_counts = {}
        self.pre_counts_ngram = {}
        self.pre_counts_ctx = {}
        self.post_counts = [{}, {}, {}]
        self.kn_discount = [0, 0, 0]

    def count(self, ngrams, hash_fn, hash_remove_first_fn, hash_remove_last_fn):
        # regular ngram counts
        self.counts += Counter(hash_fn(ngram) for ngram in ngrams)
        # counts of counts
        self.counts_of_counts = Counter(self.counts.values())
        # preceding word type counts
        self.pre_counts_ngram = Counter(hash_remove_first_fn(ngram)
            for ngram in self.counts if ngram)
        self.pre_counts_ctx = Counter()
        for ngram, c in self.pre_counts_ngram.items():
            if ngram:
                self.pre_counts_ctx[hash_remove_first_fn(ngram)] += c
        # succeeding word type counts
        for i in range(len(self.post_counts) - 1):
            self.post_counts[i] = Counter(hash_remove_last_fn(ngram)
                for ngram, c in self.counts.items() if c == i + 1)
        self.post_counts[-1] = Counter(hash_remove_last_fn(ngram)
            for ngram, c in self.counts.items() if c >= len(self.post_counts))
        # modified kneser ney discounts
        n = [self.counts_of_counts.get(i + 1, 0)
            for i in range(len(self.kn_discount) + 1)]
        d = n[0] / (n[0] + 2 * n[1])
        self.kn_discount = [(i + 1) - (i + 2) * d * n[i + 1] / n[i]
            for i in range(len(self.kn_discount))]
        # total unigram count
        null_ctx = hash_fn(None)
        self.counts[null_ctx] = sum(c for ngram, c in self.counts.items()
            if hash_remove_first_fn(ngram) == null_ctx)

class LanguageModel:
    """N-gram LM with Modified Kneser-Ney interpolated smoothing."""
    def __init__(self, order, vocab, ngram_counts=None, unk_token='<UNK>'):
        self.order = order
        self.vocab = [None, '<s>', '</s>', unk_token]
        if isinstance(vocab, str):
            with open(vocab, 'r', encoding='utf-8') as f:
                self.vocab += [line.strip() for line in f]
        elif isinstance(vocab, list):
            self.vocab = vocab
        self.vocab2id = {word: i for i, word in enumerate(self.vocab)}
        self.unk_id = self.vocab2id[unk_token]
        if ngram_counts:
            self.ngram_counts = ngram_counts
        else:
            self.ngram_counts = NgramCounts()

    def logscore(self, *args, **kwargs):
        score = self.score(*args, **kwargs)
        return math.log(score, 10) if score > 0 else float('-inf')

    def score(self, word, context=None):
        """Compute the LM probability p(e_i | e_i-1, e_i-2, ...)."""
        ctx_hash = self._ngram_hash(context)
        word_hash = self._ngram_hash(word)
        kn_score = self._kneser_ney_score(word_hash, ctx_hash)
        return kn_score

    def _kneser_ney_score(self, word_hash, ctx_hash, highest_order=True):
        if not word_hash:  # base case
            return 1 / (len(self.vocab) - 3)  # don't count special tokens
        B = len(self.vocab).bit_length()
        ngram_hash = (ctx_hash << B) + word_hash
        if highest_order:
            # use regular counts
            word_count = self.ngram_counts.counts.get(ngram_hash, 0)
            ctx_count = self.ngram_counts.counts.get(ctx_hash, 0)
        else:
            # use continuation counts
            word_count = self.ngram_counts.pre_counts_ngram.get(ngram_hash, 0)
            ctx_count = self.ngram_counts.pre_counts_ctx.get(ctx_hash, 0)
        # recursive case: discount probability and redistribute to lower order
        kn_discounts = self.ngram_counts.kn_discount
        discount = kn_discounts[min(word_count, len(kn_discounts) - 1)]
        discounted_p = max(word_count - discount, 0) / (ctx_count + 1)
        # calculate normalization weight
        n_ctxs = self.ngram_counts.post_counts
        norm_weight_numer = sum(d * n_ctx.get(ctx_hash, 0) + int(i == 0)
            for i, (d, n_ctx) in enumerate(zip(kn_discounts, n_ctxs)))
        norm_weight = norm_weight_numer / (ctx_count + 1)
        # reduce context ngram to lower order and recurse
        if ctx_hash:
            ctx_hash = self._ngram_hash_reduced(ctx_hash)
        else:
            word_hash = 0
        return discounted_p + norm_weight * self._kneser_ney_score(
            word_hash, ctx_hash, False)

    def train(self, dataset_files):
        """Trains an LM given a list of text files."""
        with fileinput.input(files=dataset_files) as f:
            ngrams, _ = padded_everygram_pipeline(self.order,
                (line.strip().split() for line in f))
            ngrams = (ngram for line in ngrams for ngram in line)
            B = len(self.vocab).bit_length()
            self.ngram_counts.count(ngrams, self._ngram_hash,
                self._ngram_hash_reduced, lambda x: x >> B)

    def _ngram_hash(self, ngram):
        """Returns hash of an ngram"""
        if not ngram:
            return 0
        if isinstance(ngram, str):
            ngram = ngram.split()
        B = len(self.vocab).bit_length()
        return sum(self.vocab2id.get(word, self.unk_id) << (B*(len(ngram)-i-1))
            for i, word in enumerate(ngram))

    def _ngram_hash_reduced(self, ngram_hash):
        """Returns hash of an ngram with its leftmost word removed."""
        B = len(self.vocab).bit_length()
        ngram_order = -(ngram_hash.bit_length() // -B)
        reduce_mask = (1 << ((ngram_order - 1) * B)) - 1
        return ngram_hash & reduce_mask

    def save(self, path):
        with open(path, 'wb') as f:
            data = {
                'order': self.order,
                'vocab': self.vocab,
                'ngram_counts': self.ngram_counts
            }
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return cls(**pickle.load(f))

def debug(args):
    lm = LanguageModel.load(args.lm_file)
    while True:
        tokens = input().strip().lower().split()
        word = tokens[-1]
        context = tokens[-lm.order:-1]
        print(lm.logscore(word, context))

def train(args):
    dataset_files = glob.glob(args.dataset_files_pattern, recursive=True)
    lm = LanguageModel(order=args.order, vocab=args.vocab_file)
    lm.train(dataset_files)
    lm.save(args.lm_file)
    print('Saved to', args.lm_file)
