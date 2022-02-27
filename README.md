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

In contrast to "regular" webspeak, we can see that Piemanese contains far more spelling perturbations, such that a simple Levenshtein distance based spelling correction algorithm or replacement dictionary is insufficient to translate it back to regular English.

## Problem Formulation
We approach this as [machine translation](https://stanford.edu/~cpiech/cs221/apps/machineTranslation.html) problem, in other words we look to compute the following:

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\arg\max_{e\in&space;E}{p(e|\pi)})

where ![E](https://latex.codecogs.com/png.image?\dpi{110}&space;E) is the set of all possible English sentences and ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) is a Piemanese sentence.

By Bayes' theorem, we can rewrite this as:

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\arg\max_{e\in&space;E}{p(\pi|e)p(e)})

We can then interpret the first term ![p(pi|e)](https://latex.codecogs.com/png.image?\dpi{110}&space;p(\pi|e)) as a **translation model** and the second term ![p(e)](https://latex.codecogs.com/png.image?\dpi{110}&space;p(e)) as a **language model**.
- **translation model**: returns a high probability if ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) is a good translation of ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e), low probability if it is not.
- **language model**: returns a high probability if ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e) is a well-formed English sentence, lower if it is not.

Then, to combine the two, we use a decoding algorithm (since it is too expensive to go through all possible English sentences). In this case, a greedy decoding algorithm is sufficient.

## Translation Model
Normally, a translation model would be a set of parameters trained on a corpus, but due to the lack of a Piemanese-English parallel corpus, we use a makeshift solution for the translation model here. Instead, we use the following function:

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;p(\pi|e)=\begin{cases}1&\pi\text{&space;is&space;a&space;valid&space;Piemanese&space;translation&space;of&space;}e\\\\0&\text{otherwise}\end{cases})

In our actual implementation, this is equivalent to computing the set:

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\\{e\in&space;E\mid&space;\pi\text{&space;is&space;a&space;valid&space;Piemanese&space;translation&space;of&space;}e\\})

### What is a "valid Piemanese translation"?
We define ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) to be a valid Piemanese translation of ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e) iff the *Piemanese root form* of ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) is equal to that of ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e). Namely, we want to compute the set:

![equation](https://latex.codecogs.com/png.image?\dpi{110}&space;\\{e\in&space;E\mid&space;root(\pi)=root(e)\\})

A *Piemanese root form* of a word is essentially **the sequence of consonant phonemes in a word**. The main rule of Piemanese appears to be that although the spelling can be distorted very far from English, the pronunciation of the words stay largely the same, with the exception of vowels.

### Replacement Dictionary
Lastly, to catch the exceptions to the rule, we also use a manually written Piemanese to English replacement dictionary before running it through the SMT pipeline. This could also be viewed as an extension of the translation model.

## Language Model
We train a trigram language model with Kneser-Ney smoothing (using NLTK modules) on the [TwitchChat](https://osf.io/39ev7/) corpus. Since we expect this translation bot to be used in a casual Discord chat, the best representation of English should not be from formal/proper English, but rather casual English seen in live chat.

The trigram language model will take the previous two words into context and select the highest probability word from our possible candidate words (selected from our makeshift "translation model"), which will disambiguate when multiple English translations are possible.
