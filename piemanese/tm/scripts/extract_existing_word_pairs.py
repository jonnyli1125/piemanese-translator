import argparse
import os
import re

def main(args):
    with open('piemanese/vocab/pi_emotes.txt', 'r') as f:
        emotes = {line.strip() for line in f}
    # extract benchmark word pairs
    with open(args.benchmark_dir + '/pi.txt', 'r', encoding='utf-8') as f:
        pi_lines = [line.strip() for line in f]
    with open(args.benchmark_dir + '/en.txt', 'r', encoding='utf-8') as f:
        en_lines = [line.strip() for line in f]
    words = []
    for pi, en in zip(pi_lines, en_lines):
        pi_words = [re.sub(r'(.)\1{2,}', r'\1\1', w) for w in pi.split() if re.sub(r'(.)\1+', r'\1', w) not in emotes and re.match(r'^[a-z0-9\']+$', w)]
        en_words = [w for w in en.split() if re.match(r'^[a-z0-9\']+$', w)]
        if len(pi_words) == 1:
            words.append((pi_words[0], ' '.join(en_words)))
            continue
        if len(pi_words) != len(en_words):
            while pi_words and en_words and len(pi_words) != len(en_words):
                if pi_words[0] == en_words[0]:
                    en_words_to_take = 1
                else:
                    print(pi_words)
                    print(en_words)
                    en_words_to_take = int(input())
                if en_words_to_take:
                    words.append((pi_words[0], ' '.join(en_words[:en_words_to_take])))
                pi_words = pi_words[1:]
                en_words = en_words[en_words_to_take:]
        words += [(p, e) for p, e in zip(pi_words, en_words)
            if re.match(r'[a-z]', p) and re.match(r'^[a-z0-9\']+$', p)]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/benchmark.tsv', 'w', encoding='utf-8') as f:
        f.writelines(f'{a}\t{b}\n' for a, b in words)

    # extract replacements word pairs
    replacements = []
    with open(args.replacements_file, 'r', encoding='utf-8') as f:
        for line in f:
            a, bs = line.strip().split('\t')
            for b in bs.split(','):
                replacements.append((a, b))
    with open(f'{args.output_dir}/replacements.tsv', 'w', encoding='utf-8') as f:
        f.writelines(f'{a}\t{b}\n' for a, b in replacements)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark_dir')
    parser.add_argument('replacements_file')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    main(args)
