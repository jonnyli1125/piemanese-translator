import argparse
import collections
import glob
import random
import json
import os
import re
import Levenshtein
from tqdm import tqdm

def main(args):
    # read english vocab
    with open(args.vocab_dir + '/vocab.txt', 'r') as f:
        vocab_words = [l.strip() for l in f]
    with open(args.vocab_dir + '/word_prob_mass.txt', 'r') as f:
        vocab_cum_weights = [float(l.split('\t')[0]) for l in f]
        vocab_cum_weights = vocab_cum_weights[:len(vocab_words)]
    # generate false pairs from mismatching existing similar en words
    false_pairs = []
    en_words = random.choices(vocab_words, cum_weights=vocab_cum_weights,
        k=len(vocab_words))
    for word in tqdm(en_words):
        word_ratios = [(Levenshtein.ratio(word, w) + random.random() * .01, w)
            for w in vocab_words
            if w.replace("'", '') != word.replace("'", '')]
        word_ratios = sorted(word_ratios, reverse=True)
        sim_words = [w for r, w in word_ratios if r > 0]
        n = args.n_repeats//2
        sim_words = set(sim_words[:n] + sim_words[-n:] + [w for w in sim_words
            if w != word and word in w][-n:])
        false_pairs += [(word, w) for w in sim_words]
    with open(args.word_pairs_dir + '/false/vocab_similar.tsv', 'w') as f:
        f.writelines(f'{en1}\t{en2}\n' for en1, en2 in false_pairs)
    # generate true pairs from punctuation correction
    true_pairs = []
    vocab_words_set = set(vocab_words)
    for word in vocab_words:
        no_apos = word.replace("'", '')
        if "'" in word and no_apos in vocab_words_set:
            true_pairs.append((no_apos, word))
    with open(args.word_pairs_dir + '/true/punctuation.tsv', 'w') as f:
        f.writelines(f'{w1}\t{w2}\n' for w1, w2 in true_pairs)
    # generate true word pairs from piemanese transformations
    # see ../notebooks/heuristics.ipynb for details
    if args.pi_edits_file:
        with open(args.vocab_dir + '/phrase_vocab.txt', 'r') as f:
            phrase_vocab = [line.strip() for line in f]
        true_pairs = []
        with open(args.pi_edits_file, 'r') as f:
            pi_edits = json.load(f)
            for edit_dict in pi_edits.values():
                for op in edit_dict:
                    a, b = op.split(',')
                    edit_dict[op] -= ((len(b) - len(a)) ** 2) * 0.1
        for word in tqdm(en_words + vocab_words + phrase_vocab):
            for i in range(args.n_repeats):
                edit = random.choices(['replace', 'delete', 'insert', 'equal'],
                    weights=[2, 1, 1, 1], k=1)[0]
                ops_in_word = [t for t in pi_edits[edit].items()
                    if t[0].split(',')[0] in word]
                if not ops_in_word:
                    continue
                ops, counts = list(zip(*ops_in_word))
                op = random.choices(ops, weights=counts, k=1)[0]
                if edit == 'equal':
                    pi_word = word
                elif edit == 'insert':
                    pi_word = word + op.split(',')[1]
                else:
                    a, b = op.split(',')
                    matches = list(re.finditer(a, word))
                    i, j = random.choice(matches).span()
                    pi_word = word[:i] + b + word[j:]
                if not pi_word:
                    continue
                true_pairs.append((pi_word, word))
        with open(args.word_pairs_dir + '/true/synthetic.tsv', 'w') as f:
            f.writelines(f'{pi}\t{en}\n' for pi, en in true_pairs)
    # TODO: generate pseudo labeled word pairs from existing model?

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('word_pairs_dir')
    parser.add_argument('vocab_dir')
    parser.add_argument('--n_repeats', type=int, default=20)
    parser.add_argument('--pi_edits_file',
        default=f'{os.path.dirname(__file__)}/pi_edits.json')
    args = parser.parse_args()

    main(args)
