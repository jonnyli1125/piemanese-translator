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

Then, we use a **decoding algorithm** (since it is too expensive to go through all possible English sentences) to combine the two models together.

## Translation Model
Normally, a translation model would consist of a set of parameters that is trained using an optimization algorithm on a parallel corpus, but since there is no Piemanese-English parallel corpus, we can't actually train our model in the traditional sense. Instead, we use an algorithmic solution for the translation model:

![equation](https://latex.codecogs.com/png.image?\dpi{110}-\log&space;p(\pi|e)=\alpha\times\text{PhonemeDistance}(\pi,e)&plus;\beta\times\text{GraphemeDistance}(\pi,e))

where ![alpha,beta](https://latex.codecogs.com/png.image?\dpi{110}\alpha,\beta&space;) are coefficients and `PhonemeDistance` is a [phonetic feature weighted Levenshtein distance](https://github.com/dmort27/panphon#the-panphondistance-module) (Mortensen et al, 2016) between the pronunciations of ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) and ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e), and `GraphemeDistance` is a grapheme based Levenshtein distance between ![pi](https://latex.codecogs.com/png.image?\dpi{110}&space;\pi) and ![e](https://latex.codecogs.com/png.image?\dpi{110}&space;e) that I defined [here](https://github.com/jonnyli1125/piemanese-translator/blob/main/translator.py).

Essentially, this results in English words that are both phonetically and graphemically similar (have less distance) to the Piemanese word to have higher probabilities than those that are not (have greater distance).

### Replacement Dictionary
To catch the exceptions, we also use a manually written Piemanese to English [replacement dictionary](https://github.com/jonnyli1125/piemanese-translator/blob/main/replacements.csv) before running it through the other components of the pipeline. This could also be viewed as an extension of the translation model.

## Language Model
We train a trigram language model with Laplace smoothing (using NLTK modules) on the [TwitchChat](https://osf.io/39ev7/) corpus.

![equation](https://latex.codecogs.com/png.image?\dpi{110}p(e)=p(e_i|e_{i-1}e_{i-2})=\frac{\text{count}(e_ie_{i-1}e_{i-2})&plus;1}{\sum_{e_k\in&space;E}{\text{count}(e_ke_{i-1}e_{i-2})&plus;1}})

Since we expect this translation bot to be used in a casual Discord chat, the best representation of English should not be from formal/proper English, but rather casual English seen in live chat.

The language model will determine the highest probability word by taking into account the context of the sentence (previous two words for a trigram model). This will help resolve ambiguous situations where a Piemanese word may have multiple valid English translations.

## Decoder
We use a greedy decoding algorithm. In our case, Piemanese is simple enough that the words are generally aligned one-to-one with regular English, so beam search decoding is not necessary.

![equation](https://latex.codecogs.com/png.image?\dpi{110}\arg\max_{e\in&space;E}{p(\pi|e)p(e)=\log&space;p(\pi|e)&plus;\log&space;p(e)})

For each word, we add the translation model log score with the language model log score for all english words given the piemanese word, and pick the one with the highest log score as our best translation.
