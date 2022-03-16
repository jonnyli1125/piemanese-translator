import argparse
import collections
import re
import dill as pickle
import phonemizer

def phonemize(text):
    """copied from translator.py"""
    text = phonemizer.phonemize(text, preserve_punctuation=True).strip()
    text = text.replace('ɚ', 'ə˞').replace('ɝ', 'ɜ˞')
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('lm_file')
    parser.add_argument('pron_file')
    args = parser.parse_args()

    with open(args.lm_file, 'rb') as f:
        lm = pickle.load(f)
    pron_lookup = collections.defaultdict(list)
    vocab = [w for w in lm.vocab if re.match(r'^[a-z]+$', w)]
    # chunk to avoid overloading phonemizer
    n_chunks = 100
    vocab_chunks = [vocab[i::n_chunks] for i in range(n_chunks)]
    pron_chunks = [phonemize(','.join(chunk)).split(',')
        for chunk in vocab_chunks]
    for vocab_chunk, pron_chunk in zip(vocab_chunks, pron_chunks):
        for word, pron in zip(vocab_chunk, pron_chunk):
            pron_lookup[pron].append(word)
    with open(args.pron_file, 'wb') as f:
        pickle.dump(pron_lookup, f)

if __name__ == '__main__':
    main()
