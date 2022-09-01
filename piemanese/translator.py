import re
import os.path
from .decoder import Decoder
from .lm import LanguageModel
from .tm import TranslationModel

class Translator:
    def __init__(self, vocab_dir=None, lm_file='lm.pkl', tm_file='tm.pkl'):
        lm = LanguageModel.load(lm_file)
        tm = TranslationModel.load(tm_file)
        self.decoder = Decoder(lm, tm)
        if not vocab_dir:
            vocab_dir = f'{os.path.dirname(__file__)}/vocab'
        with open(f'{vocab_dir}/pi_emotes.txt', 'r') as f:
            self.pi_emotes = {line.strip() for line in f}
        with open(f'{vocab_dir}/en_emotes.txt', 'r') as f:
            self.en_emotes = {line.strip() for line in f}
        self.en_phrase_repl = []
        with open(f'{vocab_dir}/en_phrase_replacements.tsv', 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                expr, repl = line.split('\t')
                self.en_phrase_repl.append((re.compile(expr), repl))

    def __call__(self, pi_sent, **kwargs):
        """Performs extra phrase replacement before and after decoding."""
        if isinstance(pi_sent, str):
            pi_tokens = self.tokenize(pi_sent)
        elif isinstance(pi_sent, list):
            pi_tokens = pi_sent
        else:
            pi_tokens = list(pi_sent)
        pi_tokens_clean = self._remove_emotes_pre(pi_tokens)
        en_tokens = self.decoder(pi_tokens_clean, **kwargs)[0][1]
        en_tokens_clean = self._remove_emotes_post(en_tokens)
        en_sent = ' '.join(en_tokens_clean)
        for phrase_re, repl_re in self.en_phrase_repl:
            en_sent = phrase_re.sub(repl_re, en_sent)
        return en_sent

    def tokenize(self, sent):
        return sent.lower().strip().split()

    def _remove_emotes_pre(self, pi_tokens):
        return [w for w in pi_tokens if w not in self.pi_emotes]

    def _remove_emotes_post(self, en_tokens):
        return [w for w in en_tokens if w not in self.en_emotes]
