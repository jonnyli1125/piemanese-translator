import argparse
import csv
import glob
import re

def preprocess(s):
    s = s.lower().strip()
    # remove bot commands
    for command_prefix in ['!', '=', '-', ';;', 'wordle']:
        if s.startswith(command_prefix):
            return ''
    # remove links
    s = re.sub(r'https?://\S+', '', s)
    # remove mentions
    s = re.sub(r'@\S+', '', s)
    return re.sub(r'\s+', ' ', s).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir')
    parser.add_argument('output_file')
    args = parser.parse_args()

    sentences = []
    for corpus_file in glob.glob(args.corpus_dir + '/*.csv'):
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                content = preprocess(row[3])
                if content:
                    sentences.append(content)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        f.writelines(s + '\n' for s in sentences)

if __name__ == '__main__':
    main()
