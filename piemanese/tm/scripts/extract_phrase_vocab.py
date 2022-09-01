import argparse

def main(args):
    phrase_vocab = set()
    for filename in ['benchmark.tsv', 'replacements.tsv']:
        with open(args.word_pairs_dir + '/true/' + filename, 'r') as f:
            for line in f:
                pi, en = line.strip().split('\t')
                if ' ' in en:
                    phrase_vocab.add(en)
    with open(args.vocab_dir + '/phrase_vocab.txt', 'w') as f:
        f.writelines(w + '\n' for w in sorted(phrase_vocab))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('word_pairs_dir')
    parser.add_argument('vocab_dir')
    args = parser.parse_args()

    main(args)
