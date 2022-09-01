import argparse

def main(args):
    words = set()
    with open(args.benchmark_dir + '/en.txt', 'r') as f:
        for line in f:
            words |= set(w.rstrip('.,!?') for w in line.strip().split())
    word_prob_mass = {}
    with open(args.word_prob_mass_file, 'r') as f:
        for line in f:
            p, word = line.split()
            word_prob_mass[word] = float(p)
    print(words - word_prob_mass.keys())
    print(sorted((word_prob_mass[w], w) for w in words & word_prob_mass.keys())[-10:])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark_dir')
    parser.add_argument('word_prob_mass_file')
    args = parser.parse_args()

    main(args)
