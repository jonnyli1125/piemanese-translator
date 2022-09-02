import argparse
import math
import glob
import os.path
from collections import Counter
from nltk.lm.preprocessing import padded_everygram_pipeline
import dill as pickle

class LanguageModel:
    """N-gram LM with Kneser-Ney smoothing."""
    def __init__(self, order=4, discount=0.1, vocab=None, ngrams=None,
            unk_token='<UNK>'):
        self.order = order
        self.discount = discount
        self.vocab = [None, '<s>', '</s>', unk_token]
        if isinstance(vocab, str):
            with open(vocab, 'r', encoding='utf-8') as f:
                self.vocab += [line.strip() for line in f]
        elif isinstance(vocab, list):
            self.vocab = vocab
        self.vocab2id = {word: i for i, word in enumerate(self.vocab)}
        self.unk_id = self.vocab2id[unk_token]
        self.ngrams = ngrams if ngrams else {}
        B = len(self.vocab).bit_length()
        self.num_ngrams_cont_word = Counter(self._ngram_hash_reduced(ngram)
            for ngram in self.ngrams if ngram)
        self.num_ngrams_cont_ctx = Counter()
        for ngram, count in self.num_ngrams_cont_word.items():
            if not ngram:
                continue
            self.num_ngrams_cont_ctx[self._ngram_hash_reduced(ngram)] += count
        self.num_ngrams_ctx = Counter(ngram >> B for ngram in self.ngrams)

    def logscore(self, word, context=None):
        """Compute LM probability p(e_i | e_i-1, e_i-2, ...)."""
        ctx_hash = self._ngram_hash(context)
        word_hash = self._ngram_hash(word)
        kn_score = self._kneser_ney_score(word_hash, ctx_hash)
        return math.log(kn_score) if kn_score > 0 else -99

    def _kneser_ney_score(self, word_hash, ctx_hash, highest_order=True):
        if not word_hash:  # base case
            return 1 / (len(self.vocab) - 3)  # don't count special tokens
        B = len(self.vocab).bit_length()
        ngram_hash = (ctx_hash << B) + word_hash
        if highest_order:
            # use regular counts
            word_count = self.ngrams.get(ngram_hash, 0)
            ctx_count = self.ngrams.get(ctx_hash, 0)
        else:
            # use continuation counts
            word_count = self.num_ngrams_cont_word.get(ngram_hash, 0)
            ctx_count = self.num_ngrams_cont_ctx.get(ctx_hash, 0)
        # recursive case: discount probability and redistribute to lower order
        discounted_p = max(word_count - self.discount, 0) / (ctx_count + 1)
        # calculate normalization weight
        n_words_in_ctx = self.num_ngrams_ctx.get(ctx_hash, 0)
        norm_weight = (self.discount * n_words_in_ctx + 1) / (ctx_count + 1)
        # reduce context ngram to lower order and recurse
        if ctx_hash:
            ctx_hash = self._ngram_hash_reduced(ctx_hash)
        else:
            word_hash = 0
        return discounted_p + norm_weight * self._kneser_ney_score(
            word_hash, ctx_hash, False)

    def train(self, dataset_files):
        """Trains an LM given a list of text files."""
        datasets = [self._train_dataset(dataset_file)
            for dataset_file in dataset_files if os.path.isfile(dataset_file)]
        self.ngrams = self._combine_datasets(datasets)
        # TODO spark/mapreduce?

    def _combine_datasets(self, datasets, scale_weight=False):
        max_weight = max(w for w, ngrams in datasets)
        combined_ngrams = Counter()
        for dataset_weight, dataset_ngrams in datasets:
            if scale_weight:
                weight = int(max_weight / dataset_weight)
            else:
                weight = 1
            for ngram, count in dataset_ngrams.items():
                combined_ngrams[ngram] += weight * count
        return dict(combined_ngrams)

    def _train_dataset(self, dataset_file):
        with open(dataset_file, 'r', encoding='utf-8') as f:
            weight = sum(len(line.strip().split()) for line in f)
        with open(dataset_file, 'r', encoding='utf-8') as f:
            lines = [line.strip().split() for line in f]
            ngrams, _ = padded_everygram_pipeline(self.order, lines)
            ngrams_dict = Counter(self._ngram_hash(ngram)
                for line in ngrams for ngram in line)
            ngrams_dict[0] = sum(ngrams_dict.values())
        return weight, ngrams_dict

    def _ngram_hash(self, ngram):
        if not ngram:
            return 0
        if isinstance(ngram, str):
            ngram = ngram.split()
        B = len(self.vocab).bit_length()
        return sum(self.vocab2id.get(word, self.unk_id) << (B*(len(ngram)-i-1))
            for i, word in enumerate(ngram))

    def _ngram_hash_reduced(self, ngram_hash):
        B = len(self.vocab).bit_length()
        ngram_order = -(ngram_hash.bit_length() // -B)
        reduce_mask = (1 << ((ngram_order - 1) * B)) - 1
        return ngram_hash & reduce_mask

    def save(self, path):
        with open(path, 'wb') as f:
            data = {
                'order': self.order,
                'discount': self.discount,
                'vocab': self.vocab,
                'ngrams': self.ngrams
            }
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return cls(**pickle.load(f))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_files_pattern')
    parser.add_argument('lm_file')
    parser.add_argument('--order', default=4)
    parser.add_argument('--vocab_file',
        default=f'{__file__}/../../vocab/vocab.txt')
    args = parser.parse_args()

    dataset_files = glob.glob(args.dataset_files_pattern, recursive=True)
    lm = LanguageModel(args.order, vocab=args.vocab_file)
    lm.train(dataset_files)
    lm.save(args.lm_file)
