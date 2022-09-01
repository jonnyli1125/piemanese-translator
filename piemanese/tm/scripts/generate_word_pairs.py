import argparse
import collections
import glob
import random
import Levenshtein
from tqdm import tqdm

def main(args):
    # read existing true pairs
    pi_words = collections.defaultdict(set)
    en_words = []
    for word_pairs_file in ['benchmark.tsv', 'replacements.tsv']:
        with open(args.word_pairs_dir + '/true/' + word_pairs_file, 'r') as f:
            for line in f:
                pi_word, en_word = line.strip().split('\t')
                pi_words[pi_word].add(en_word)
                en_words.append(en_word)
    # read english vocab
    with open(args.vocab_dir + '/vocab.txt', 'r') as f:
        vocab_words = [l.strip() for l in f]
    with open(args.vocab_dir + '/word_prob_mass.txt', 'r') as f:
        vocab_cum_weights = [float(l.split('\t')[0]) for l in f]
        vocab_cum_weights = vocab_cum_weights[:len(vocab_words)]
    # generate false pairs from mismatching existing pi/en pairs
    false_pairs = []
    for pi_word, pi2en_words in tqdm(pi_words.items()):
        other_en_words = [w for w in en_words if w not in pi2en_words]
        n_repeats_vocab = max(1, int(random.gauss(
            len(other_en_words)/10, args.n_repeats/10)))
        n_repeats_existing = args.n_repeats - n_repeats_vocab
        random_en_words = random.choices(other_en_words, k=n_repeats_existing)
        random_en_words += random.choices(vocab_words,
            cum_weights=vocab_cum_weights, k=n_repeats_vocab)
        false_pairs += [(pi_word, w) for w in random_en_words]
    true_pairs_dir = args.word_pairs_dir + '/true'
    with open(args.word_pairs_dir + '/false/from_pi.tsv', 'w') as f:
        f.writelines(f'{pi}\t{en}\n' for pi, en in false_pairs)
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
        n = args.n_repeats//100
        sim_words = set(sim_words[:n] + sim_words[-n:] + [w for w in sim_words
            if w != word and word in w][-n:])
        false_pairs += [(word, w) for w in sim_words]
    with open(args.word_pairs_dir + '/false/vocab_similar.tsv', 'w') as f:
        f.writelines(f'{en1}\t{en2}\n' for en1, en2 in false_pairs)
    # generate true pairs from duplicating en words
    with open(args.word_pairs_dir + '/true/vocab.tsv', 'w') as f:
        f.writelines(f'{w}\t{w}\n' for w in en_words)
        f.writelines(f'{w}\t{w}\n' for w in vocab_words)
    # generate true pairs from punctuation correction
    true_pairs = []
    vocab_words_set = set(vocab_words)
    for word in vocab_words:
        no_apos = word.replace("'", '')
        if "'" in word and no_apos in vocab_words_set:
            true_pairs.append((no_apos, word))
    with open(args.word_pairs_dir + '/true/punctuation.tsv', 'w') as f:
        f.writelines(f'{w1}\t{w2}\n' for w1, w2 in true_pairs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('word_pairs_dir')
    parser.add_argument('vocab_dir')
    parser.add_argument('--n_repeats', type=int, default=1000)
    args = parser.parse_args()

    main(args)