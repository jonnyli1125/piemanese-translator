import argparse
from piemanese.tm import TranslationModel

def main(args):
    tm = TranslationModel.load(args.tm_file)
    while True:
        word = input().strip().lower()
        scores = tm.logscores(word)
        print(sorted(scores.items(), key=lambda x: -x[1]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tm_file')
    args = parser.parse_args()

    main(args)
