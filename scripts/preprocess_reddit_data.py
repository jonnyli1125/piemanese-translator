import argparse
import re
import string
import os
import glob
import unidecode
import nltk
import emoji

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
punct = r'!"#$%()*+,-./:;<=>?@[\\]^_`{|}~'
punct_dict = str.maketrans(punct, ' ' * len(punct))
url_re = re.compile(r'http\S+')
space_re = re.compile(r'\s+')
tag_re = re.compile(r'(^|\s)[#\$@/*]\w+')
word_re = re.compile(r'^[a-z]\D*$')
reddit_re = re.compile(r'(r|u)/\w+')
repeat_chars_re = re.compile(r'([a-z])\1{2,}')

def clean(line):
    line = line.strip().lower()
    line = unidecode.unidecode(line)
    line = emoji.replace_emoji(line, replace='')
    line = url_re.sub('', line)
    line = tag_re.sub('', line)
    line = reddit_re.sub('', line)
    line = repeat_chars_re.sub(r'\1', line)
    line = line.replace(' & ', ' and ')
    line = line.replace('[removed]', '').replace('[deleted]', '')
    sents = []
    for sent in sent_tokenizer.tokenize(line):
        sent = sent.translate(punct_dict)
        words = [word.strip("'") for word in sent.split()]
        words = [word for word in words if word_re.match(word)]
        sent = ' '.join(word for word in words)
        if sent:
            sents.append(sent)
    return sents

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    files = glob.glob('**/*.txt', root_dir=args.input_dir, recursive=True)
    for file in files:
        input_file = os.path.join(args.input_dir, file)
        with open(input_file, 'r', encoding='utf-8') as f:
            cleaned_lines = [sent for line in f for sent in clean(line)]
        output_file = os.path.join(args.output_dir, file)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(f'{line}\n' for line in cleaned_lines)
        print(f'Wrote {len(cleaned_lines)} lines to {output_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    main(args)
