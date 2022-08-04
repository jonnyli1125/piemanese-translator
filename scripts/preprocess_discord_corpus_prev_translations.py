import argparse
import csv
import glob
import re
import os

def preprocess(s):
    s = s.lower().strip()
    # remove bot commands
    for command_prefix in ['!', '=', '-', ';;', 'wordle']:
        if s.startswith(command_prefix):
            return ''
    if 'semantle' in s or 'letterle' in s:
        return ''
    # remove links
    s = re.sub(r'https?://\S+', '', s)
    # remove mentions
    s = re.sub(r'@\S+', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir')
    parser.add_argument('--output_dir', default='./')
    args = parser.parse_args()

    pi_sents = []
    en_sents = []
    pi_corpus = []
    for corpus_file in glob.glob(args.corpus_dir + '/*.csv'):
        with open(corpus_file, 'r', encoding='utf-8') as f:
            prev_user = None
            prev_sent = None
            cur_user = None
            for row in csv.reader(f):
                cur_user = row[1]
                cur_sent = preprocess(row[3])
                if cur_sent:
                    if cur_user.startswith('piemanese-translator') and prev_user.startswith('Pieman778'):
                        pi_sents.append(prev_sent)
                        en_sents.append(cur_sent)
                    if cur_user.startswith('Pieman778'):
                        pi_corpus.append(cur_sent)
                prev_user = cur_user
                prev_sent = cur_sent
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'pi.txt'), 'w', encoding='utf-8') as f:
        f.writelines(sent + '\n' for sent in pi_sents)
    with open(os.path.join(args.output_dir, 'en.txt'), 'w', encoding='utf-8') as f:
        f.writelines(sent + '\n' for sent in en_sents)
    with open(os.path.join(args.output_dir, 'corpus.txt'), 'w', encoding='utf-8') as f:
        f.writelines(sent + '\n' for sent in pi_corpus)

if __name__ == '__main__':
    main()
