import argparse
import glob
import re
import h5py
from transformers import BertTokenizerFast
from tqdm import tqdm

def tokenize_and_write(tokenizer, new_words, data, C):
    rows = tokenizer(
            new_words,
            return_tensors='np',
            padding='max_length',
            truncation=True,
            max_length=C).input_ids
    data.resize((data.shape[0] + len(rows), C))
    data[-len(rows):] = rows

def main(args):
    batch_size = 10000
    C = args.columns

    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_dir)

    hf = h5py.File(args.output_file, 'w')
    data = hf.create_dataset('data', (0, C), maxshape=(None, C), dtype='i2')
    new_words = []
    for file in tqdm(glob.glob(f'{args.input_dir}/*')):
        n = 0
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().lower().split()
                for word in words:
                    word = re.sub(r'[?!.,]+$', '', word)
                    if not re.match(r"^[a-z0-9']*[a-z][a-z0-9']*$", word):
                        continue
                    word = re.sub(r"[^a-z0-9']", '', word)
                    word = re.sub(r'([a-z])\1{2,}', r'\1\1', word)
                    word = ' '.join(word)
                    new_words.append(word)
                if len(new_words) >= batch_size:
                    tokenize_and_write(tokenizer, new_words, data, C)
                    new_words.clear()
                n += 1
                if n >= args.max_lines_per_file:
                    break
    if new_words:
        tokenize_and_write(tokenizer, new_words, data, C)
    print(data.shape[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file')
    parser.add_argument('input_dir')
    parser.add_argument('tokenizer_dir')
    parser.add_argument('--columns', default=20, type=int)
    parser.add_argument('--max_lines_per_file', default=10000, type=int)
    args = parser.parse_args()

    main(args)
