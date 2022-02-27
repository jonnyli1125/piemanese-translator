import argparse
import re
import glob
import nltk
from nltk.lm import KneserNeyInterpolated, Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline
import dill as pickle

def condition(text):
    text = text.split(',', 3)[-1]
    tokens = text.split()
    return [t for t in tokens if re.match(r"^[\w'-]{1,39}$", t)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir')
    parser.add_argument('model_file')
    parser.add_argument('-o', '--order', default=3, type=int)
    args = parser.parse_args()

    corpus = []
    for corpus_file in glob.glob(f'{args.corpus_dir}/*.*'):
        with open(corpus_file, 'r', encoding='utf-8') as f:
            corpus += [condition(line) for line in f]
    corpus = [line for i, line in enumerate(corpus) if i % 50 == 0]
    print(len(corpus))

    train, vocab = padded_everygram_pipeline(args.order, corpus)
    vocab = Vocabulary(vocab, unk_cutoff=20)
    model = KneserNeyInterpolated(args.order, vocabulary=vocab)
    model.fit(train)

    with open(args.model_file, 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    main()
