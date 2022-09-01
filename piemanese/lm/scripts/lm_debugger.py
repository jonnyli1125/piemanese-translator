import argparse
from piemanese.lm import LanguageModel

def main(args):
    lm = LanguageModel.load(args.lm_file)
    while True:
        tokens = input().strip().lower().split()
        word = tokens[-1]
        context = tokens[-lm.order:-1]
        print(lm.logscore(word, context))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lm_file')
    args = parser.parse_args()

    main(args)
