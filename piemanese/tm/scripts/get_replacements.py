import collections
import os.path

def read_corpus(corpus_file='piemanese/tm/datasets/piemanese/corpus.txt'):
    with open(corpus_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def word_freq(corpus):
    freq = collections.Counter()
    for line in corpus:
        freq.update(line.split())
    return freq

def main():
    corpus = read_corpus()
    freq = word_freq(corpus)
    kv_sorted = sorted(freq.items(), key=lambda x: -x[1])
    with open('piemanese/tm/datasets/piemanese/freq.txt', 'w', encoding='utf-8') as f:
        f.writelines(f'{k} {v}\n' for k, v in kv_sorted)
    if os.path.exists('replacements.tsv'):
        print('replacements file already exists')
        return
    with open('replacements.tsv', 'w', encoding='utf-8') as f:
        for k, v in kv_sorted:
            if v != 1:
                continue
            print(k)
            replacement = input('replacement: ')
            if replacement:
                f.write(f'{k}\t{replacement}\n')

if __name__ == '__main__':
    main()
