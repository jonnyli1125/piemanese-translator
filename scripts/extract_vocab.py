import argparse
import glob
import os
import math
from collections import Counter, defaultdict

def main(args):
    # count individual dataset frequencies
    input_files = glob.glob(f'{args.input_dir}/**/*.txt', recursive=True)
    unigram_lms = {}
    for input_file in input_files:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]
            freq = Counter(word for line in lines for word in line.split())
            unigram_lms[input_file] = [len(lines), freq]

    # combine dataset frequencies
    max_weight = max(w for w, c in unigram_lms.values())
    unigram_prob = defaultdict(int)
    for input_file in input_files:
        weight = float(max_weight) / unigram_lms[input_file][0]
        freq = unigram_lms[input_file][1]
        for word, count in freq.items():
            unigram_prob[word] += weight * count

    # get cumulative word probabilty mass
    z = sum(unigram_prob.values())
    for word in unigram_prob:
        unigram_prob[word] /= z
    cum_p = 0
    word_prob_mass = []
    for w, p in sorted(unigram_prob.items(), key=lambda x: -x[1]):
        cum_p += p
        word_prob_mass.append((cum_p, w))

    # write to output files
    os.makedirs(args.output_dir, exist_ok=True)
    word_prob_mass_file = f'{args.output_dir}/word_prob_mass.txt'
    with open(word_prob_mass_file, 'w', encoding='utf-8') as f:
        f.writelines(f'{p}\t{w}\n' for p, w in word_prob_mass)
    max_p = float(args.max_prob)
    vocab_file = f'{args.output_dir}/vocab.txt'
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.writelines(f'{w}\n' for p, w in word_prob_mass if p <= max_p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--max_prob', default=1.)
    args = parser.parse_args()

    main(args)
