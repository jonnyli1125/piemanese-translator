import argparse
import os.path
from difflib import SequenceMatcher
from .translator import Translator

def main(args):
    if not args.benchmark_dir:
        args.benchmark_dir = f'{os.path.dirname(__file__)}/benchmark'
    with open(args.benchmark_dir + '/pi.txt', 'r', encoding='utf-8') as f:
        pi_lines = [line.strip() for line in f.readlines()]
    with open(args.benchmark_dir + '/en.txt', 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f.readlines()]
    translator = Translator()
    words = 0
    words_err = 0
    sents = len(en_lines)
    sents_err = 0
    print('\t'.join(['pi', 'en_true', 'en_pred', 'errors']))
    for pi, en_true in zip(pi_lines, en_lines):
        en_pred = translator(pi, verbose=args.verbose)
        en_pred_words = en_pred.split()
        en_true_words = en_true.split()
        words += len(en_true_words)
        s = SequenceMatcher(None, en_pred_words, en_true_words)
        errors = sum(
            max(i2 - i1, j2 - j1)
            for tag, i1, i2, j1, j2 in s.get_opcodes()
            if tag != 'equal')
        if errors:
            words_err += errors
            sents_err += 1
        if not args.errors_only or errors:
            print('\t'.join([pi, en_true, en_pred, str(errors)]))
    print(f'WER: {words_err}/{words} ({words_err/words*100}%)')
    print(f'SER: {sents_err}/{sents} ({sents_err/sents*100}%)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark_dir')
    parser.add_argument('-e', '--errors_only', action='store_true')
    parser.add_argument('-v', '--verbose', default=0, type=int, choices=[0,1,2])
    args = parser.parse_args()

    main(args)
