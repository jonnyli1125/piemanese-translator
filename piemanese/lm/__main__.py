import argparse
import os.path
from .lm import train, debug

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.set_defaults(fn=None)
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train_parser.add_argument('dataset_files_pattern')
    train_parser.add_argument('--lm_file', default='lm.pkl', help='Name of LM file')
    train_parser.add_argument('--order', default=3, type=int, help='Max ngram order')
    train_parser.add_argument('--vocab_file', default=f'{os.path.dirname(__file__)}/../vocab/vocab.txt', help='Path to vocab file')
    train_parser.set_defaults(fn=train)

    debug_parser = subparsers.add_parser('debug', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    debug_parser.add_argument('--lm_file', default='lm.pkl', help='Name of LM file')
    debug_parser.set_defaults(fn=debug)

    args = parser.parse_args()
    if args.fn:
        args.fn(args)
    else:
        parser.print_usage()
