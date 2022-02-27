# Piemanese (Webspeak) to English Translator
Simple webspeak to English SMT model as a Discord bot.

The bot can be run with `python3 bot.py` after assigning proper values to `config.json`.

## What is Piemanese?
Piemanese, is a form of webspeak spoken by my friend Pieman.

Some examples of Piemanese (First line Piemanese, English below):
```
i ges i cn liftu a beet ;-;
i guess i can lift a bit ;-;

i told u to pley it b4 >.<
i told you to play it before >.<

mai englando es too gud
my english is too good
```

Furthermore, some Piemanese words can be ambiguous and need to be determined by context.

Example of an ambiguous case: `wan`
```
wan u come
when you come

nani u wan
what you want
```

In contrast to "regular" webspeak, we can see that Piemanese contains far more spelling perturbations, such that a simple Levenshtein distance based spelling correction algorithm or replacement dictionary is insufficient to translate it back to regular English.

A more sophisticated approach is required; one that takes into account the following:
1. How the spelling of a Piemanese word relates to its corresponding English word
2. Context of the sentence

## Problem Formulation
We approach this as [machine translation](https://stanford.edu/~cpiech/cs221/apps/machineTranslation.html) problem, in other words we look to compute the following:

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\arg\max_{e\in&space;E}{p(e|\pi)})

where ![E](https://latex.codecogs.com/png.image?\dpi{110}&space;E) is the set of all possible English sentences and ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) is a Piemanese sentence.

By Bayes' theorem, we can rewrite this as:

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\arg\max_{e\in&space;E}{p(\pi|e)p(e)})

We can then interpret the first term ![p(pi|e)](https://latex.codecogs.com/png.image?\dpi{110}&space;p(\pi|e)) as a **translation model** and the second term ![p(e)](https://latex.codecogs.com/png.image?\dpi{110}&space;p(e)) as a **language model**.
- **translation model**: returns a high probability if ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) is a good translation of ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e), low probability if it is not.
- **language model**: returns a high probability if ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e) is a well-formed English sentence, lower if it is not.

Then, we use a decoding algorithm (since it is too expensive to go through all possible English sentences) to combine the two models together. In this case, a greedy decoding algorithm is sufficient. Piemanese is simple enough that the words are aligned one-to-one with regular English, so beam search decoding is not necessary.

## Translation Model
Normally, a translation model would consist of a set of parameters that is trained using an optimization algorithm on a parallel corpus, but since there is no Piemanese-English parallel corpus, we can't actually train our model in the traditional sense. Instead, we use the following function as a makeshift solution for the translation model:

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;p(\pi|e)=\begin{cases}1&\pi\text{&space;is&space;a&space;valid&space;Piemanese&space;translation&space;of&space;}e\\\\0&\text{otherwise}\end{cases})

### What is a "valid Piemanese translation"?
We define ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) to be a valid Piemanese translation of ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e) iff the *Piemanese root form* of ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) is equal to that of ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e).

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;p(\pi|e)=\begin{cases}1&PiemaneseRoot(\pi)=PiemaneseRoot(e)\\\\0&otherwise\end{cases})

A *Piemanese root form* of a word is essentially **the sequence of consonant phonemes in a word**. The main rule of Piemanese appears to be that although the spelling can be distorted very far from English, the pronunciation of the words stay largely the same, with the exception of vowels.

For example, for this Piemanese-English pair of words:
```
mite
might
```
The Piemanese root form (sequence of consonant phonemes) of both of them is `/m t/`, so therefore `mite` is a valid Piemanese translation of `might`. The result is that for the English word `might` (and every other English word that has the root form `/m t/`), we assign probability of `mite` as 1, and other Piemanese words that don't have the same root as 0.

### Replacement Dictionary
To catch the exceptions to the rule, we also use a manually written Piemanese to English replacement dictionary before running it through the SMT pipeline. This could also be viewed as an extension of the translation model.

## Language Model
We train a trigram language model with Kneser-Ney smoothing (using NLTK modules) on the [TwitchChat](https://osf.io/39ev7/) corpus. Since we expect this translation bot to be used in a casual Discord chat, the best representation of English should not be from formal/proper English, but rather casual English seen in live chat.

The language model will determine the highest probability word from our possible candidate words (selected from our makeshift translation model) by taking into account the context of the sentence. This will resolve ambiguous situations where multiple English translations are possible.

Continuing our example from above, if we see the Piemanese word `mite`, we can compute the set of English words such that `mite` is a valid Piemanese translation of that word, using the translation model:
```
['might', 'meat', 'meet', 'met', 'matt']
```
Then, we can determine which of these English words has the highest probability of occuring given the context of the sentence. In this case, a trigram language model means we take the previous two words into account.

If we consider the sentence `maybe i mite`, if `might` was the highest probability word given the context `maybe i`, i.e.

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;p(\text{maybe&space;i&space;might})>p(\text{maybe&space;i&space;meat}),p(\text{maybe&space;i&space;meet}),...)

Then we choose `might` as our translation of `mite`.
