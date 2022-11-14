import argparse
import math
import glob
import fileinput
from collections import Counter
from nltk.lm.preprocessing import padded_everygram_pipeline
import dill as pickle

class ModifiedKneserNey:
    """
    Compute n-gram counts, probabilities, backoff weights using Modified
    Kneser-Ney smoothing.
    """
    def __init__(self, hash_fn, hash_remove_first_fn, hash_remove_last_fn,
            vocab_size):
        self.counts = Counter()
        self.counts_of_counts = {}
        self.pre_counts_ngram = {}
        self.pre_counts_ctx = {}
        self.post_counts = [{}, {}, {}]
        self.kn_discount = [0, 0, 0]
        self.hash_fn = hash_fn
        self.hash_remove_first_fn = hash_remove_first_fn
        self.hash_remove_last_fn = hash_remove_last_fn
        self.vocab_size = vocab_size

    def count(self, ngrams):
        # regular ngram counts
        self.counts += Counter(self.hash_fn(ngram) for ngram in ngrams)
        # counts of counts
        self.counts_of_counts = Counter(self.counts.values())
        # preceding word type counts
        self.pre_counts_ngram = Counter(self.hash_remove_first_fn(ngram)
            for ngram in self.counts if ngram)
        self.pre_counts_ctx = Counter()
        for ngram, c in self.pre_counts_ngram.items():
            if ngram:
                self.pre_counts_ctx[self.hash_remove_last_fn(ngram)] += c
        # succeeding word type counts
        for i in range(len(self.post_counts) - 1):
            self.post_counts[i] = Counter(self.hash_remove_last_fn(ngram)
                for ngram, c in self.counts.items() if c == i + 1)
        self.post_counts[-1] = Counter(self.hash_remove_last_fn(ngram)
            for ngram, c in self.counts.items() if c >= len(self.post_counts))
        # modified kneser ney discounts
        n = [self.counts_of_counts.get(i + 1, 0)
            for i in range(len(self.kn_discount) + 1)]
        d = n[0] / (n[0] + 2 * n[1])
        self.kn_discount = [(i + 1) - (i + 2) * d * n[i + 1] / n[i]
            for i in range(len(self.kn_discount))]
        # total unigram count
        null_ctx = self.hash_fn(None)
        self.counts[null_ctx] = sum(c for ngram, c in self.counts.items()
            if self.hash_remove_first_fn(ngram) == null_ctx)

    def prob_backoff(self):
        prob = {}
        backoff = {}
        for ngram_hash in self.counts:
             prob[ngram_hash] = self.kneser_ney_prob(ngram_hash)
             if ngram_hash in self.post_counts:
                 backoff[ngram_hash] = self.kneser_ney_backoff(ngram_hash,
                    self.counts[ngram_hash])
        return prob, backoff

    def kneser_ney_prob(self, ngram_hash, highest_order=True):
        if not ngram_hash:  # base case
            return 1 / self.vocab_size
        ctx_hash = self.hash_remove_last_fn(ngram_hash)
        if highest_order:
            # use regular counts
            word_count = self.counts.get(ngram_hash, 0)
            ctx_count = self.counts.get(ctx_hash, 0)
        else:
            # use continuation counts
            word_count = self.pre_counts_ngram.get(ngram_hash, 0)
            ctx_count = self.pre_counts_ctx.get(ctx_hash, 0)
        # recursive case: discount probability and redistribute to lower order
        discount = self.kn_discount[min(word_count, len(self.kn_discount)) - 1]
        discounted_p = max(word_count - discount, 0) / ctx_count
        # calculate interpolation weight
        int_weight = self.kneser_ney_backoff(ctx_hash, highest_order)
        # back off to lower order and recurse
        backoff_hash = self.hash_remove_first_fn(ngram_hash)
        return discounted_p + int_weight * self.kneser_ney_prob(
            backoff_hash, False)

    def kneser_ney_backoff(self, ctx_hash, highest_order=True):
        numer = sum(d * n_ctx.get(ctx_hash, 0)
            for i, (d, n_ctx) in enumerate(zip(self.kn_discount,
            self.post_counts)))
        if highest_order:
            return numer / self.counts[ctx_hash]
        else:
            return numer / self.pre_counts_ctx[ctx_hash]


class LanguageModel:
    """N-gram LM with backoff."""
    def __init__(self, order, vocab, unk_token='<UNK>', prob={}, backoff={}):
        self.order = order
        self.vocab = [None, '<s>', '</s>', unk_token]
        if isinstance(vocab, str):
            with open(vocab + '/vocab.txt', 'r', encoding='utf-8') as f:
                self.vocab += [line.strip() for line in f]
            with open(vocab + '/names.txt', 'r', encoding='utf-8') as f:
                self.vocab += [line.strip() for line in f]
            self.vocab = list(dict.fromkeys(self.vocab))
        elif isinstance(vocab, list):
            self.vocab = vocab
        self.vocab2id = {word: i for i, word in enumerate(self.vocab)}
        self.unk_id = self.vocab2id[unk_token]
        self.prob = prob
        self.backoff = backoff

    def logscore(self, *args, **kwargs):
        score = self.score(*args, **kwargs)
        return math.log(score, 10) if score > 0 else float('-inf')

    def score(self, word, context=None):
        """Compute the LM probability p(e_i | e_i-1, e_i-2, ...)."""
        B = len(self.vocab).bit_length()
        ctx_hash = self._ngram_hash(context)
        word_hash = self._ngram_hash(word)
        ngram_hash = (ctx_hash << B) + word_hash
        return self.backoff_score(ngram_hash)

    def backoff_score(self, ngram_hash):
        B = len(self.vocab).bit_length()
        ctx_hash = ngram_hash >> B
        if ngram_hash in self.prob:
            return self.prob[ngram_hash]
        else:
            backoff_weight = self.backoff.get(ctx_hash, 1)
            backoff_hash = self._ngram_hash_reduced(ngram_hash)
            return backoff_weight * self.backoff_score(backoff_hash)

    def train(self, dataset_files):
        """Trains an LM given a list of text files."""
        with fileinput.input(files=dataset_files) as f:
            ngrams, _ = padded_everygram_pipeline(self.order,
                (line.strip().split() for line in f))
            ngrams = (ngram for line in ngrams for ngram in line)
            B = len(self.vocab).bit_length()
            mkn = ModifiedKneserNey(self._ngram_hash,
                self._ngram_hash_reduced,
                lambda x: x >> B,
                len(self.vocab) - 3)
            mkn.count(ngrams)
            self.prob, self.backoff = mkn.prob_backoff()

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
                'prob': self.prob,
                'backoff': self.backoff
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
    lm = LanguageModel(order=args.order, vocab=args.vocab_dir)
    lm.train(dataset_files)
    lm.save(args.lm_file)
    print('Saved to', args.lm_file)
