import argparse
from difflib import SequenceMatcher
from translator import Translator

def main(args):
    with open(args.pi_file, 'r', encoding='utf-8') as f:
        pi_lines = [line.strip() for line in f.readlines()]
    with open(args.en_file, 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f.readlines()]
    translator = Translator()
    words = 0
    words_err = 0
    sents = len(en_lines)
    sents_err = 0
    print('\t'.join(['pi', 'en_true', 'en_pred', 'errors']))
    for pi, en_true in zip(pi_lines, en_lines):
        en_pred = translator(pi)
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
        print('\t'.join([pi, en_true, en_pred, str(errors)]))
    print(f'WER: {words_err}/{words} ({words_err/words*100}%)')
    print(f'SER: {sents_err}/{sents} ({sents_err/sents*100}%)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pi_file', help='Reference Piemanese file')
    parser.add_argument('en_file', help='Reference English file')
    args = parser.parse_args()

    main(args)
