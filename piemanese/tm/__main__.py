import argparse
from .tm import train, debug

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.set_defaults(fn=None)
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--tm_file', default='tm.pkl', help='Name of TM file')
    train_parser.set_defaults(fn=train)

    debug_parser = subparsers.add_parser('debug')
    debug_parser.add_argument('--tm_file', default='tm.pkl', help='Name of TM file')
    debug_parser.set_defaults(fn=debug)

    args = parser.parse_args()
    if args.fn:
        args.fn(args)
    else:
        parser.print_usage()
