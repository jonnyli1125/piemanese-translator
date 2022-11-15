import re
import math

class Decoder:
    def __init__(self, lm, tm):
        self.lm = lm
        self.tm = tm

    def _split_punctuation(self, word):
        """Splits a token into word, punctuation"""
        match = re.match(r"^([a-z][a-z0-9']*)([?.!,]+)$", word)
        if match:
            return match.group(1), match.group(2)
        else:
            return word, ''

    def __call__(self, pi_tokens, verbose=0, n=4):
        """
        Get top n English translations of sentence by beam search decoding.
        In mathematical terms this is:
            argmax_e p(pi|e) * p(e)
            = argmax_e p(e|pi)

        returns: [(log_prob, en_tokens)]
        n: number of beams.
        verbose=0: print nothing.
        verbose=1: show top n sentences at each decoding step.
        verbose=2: also show tm and lm scores at each step.
        """
        pi_tokens = ['<s>'] + pi_tokens + ['</s>']
        topn_sents = [(0, [])]
        pi_tokens_word = []
        pi_tokens_punc = []
        for pi_token in pi_tokens:
            word, punc = self._split_punctuation(pi_token)
            pi_tokens_word.append(word)
            pi_tokens_punc.append(punc)
        pi_tokens_word = self.tm.clean_words(pi_tokens_word)
        tm_scores_all = self.tm.multiple_scores(pi_tokens_word, top_n=n)
        for i, (word, punc) in enumerate(zip(pi_tokens_word, pi_tokens_punc)):
            tm_scores = tm_scores_all[word]
            if not tm_scores:
                continue
            new_topn_sents = []
            for p, en_tokens in topn_sents:
                combined_scores = {}
                lm_scores = {}
                for tm_word, tm_score in tm_scores.items():
                    lm_score = 1
                    tm_word_tokens = tm_word.split()
                    context = en_tokens[-self.lm.order+1:]
                    context = [self._split_punctuation(t)[0] for t in context]
                    for w in tm_word_tokens:
                        lm_score *= self.lm.score(w, context)
                        if len(context) == self.lm.order-1:
                            context = context[1:] + [w]
                        else:
                            context.append(w)
                    lm_scores[tm_word] = lm_score
                combined_scores = self._interpolate_scores(tm_scores, lm_scores)
                topn_words = sorted(combined_scores.items(),
                    key=lambda x: -x[1])[:n]
                if verbose >= 2:
                    print([(w, p, tm_scores[w], lm_scores[w])
                        for w, p in topn_words])
                for tm_word, combined_score in topn_words:
                    new_p = p + combined_score
                    new_tokens = en_tokens + (tm_word + punc).split()
                    new_topn_sents.append((new_p, new_tokens))
            topn_sents = sorted(new_topn_sents,
                key=lambda x: -x[0] / len(x[1]))[:n]
            if verbose:
                print(topn_sents)
        return [(log_p, en_tokens[1:-1]) for log_p, en_tokens in topn_sents]

    def _interpolate_scores(self, tm_scores, lm_scores):
        # TODO find better way to interpolate tm/lm lm_scores
        # upweight tm score if lm probs are all low etc
        tm_scores = self._log_normalize_scores(tm_scores)
        lm_scores = self._log_normalize_scores(lm_scores)
        combined_scores = {}
        for word in tm_scores:
            combined_scores[word] = tm_scores[word] + lm_scores[word]
        return combined_scores

    def _log_normalize_scores(self, scores):
        z = max(sum(scores.values()), 1)
        log_normalized_scores = {w: math.log(p / z, 10) if p > 0 else -99
            for w, p in scores.items()}
        return log_normalized_scores
