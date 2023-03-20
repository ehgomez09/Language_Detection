import typing
import math
import nltk
import collections
from nltk.corpus import udhr 
nltk.download('udhr') # udhr = Universal Declaration of Human Rights


print(f"There are {len(udhr.fileids())} files with the following ids: {udhr.fileids()}")

languages = ['english', 'german', 'dutch', 'french', 'italian', 'spanish']
language_ids = ['English-Latin1', 'German_Deutsch-Latin1', 'Dutch_Nederlands-Latin1', 'French_Francais-Latin1', 'Italian_Italiano-Latin1', 'Spanish_Espanol-Latin1']# I chose the above sample of languages as they all use similar characters. 

def extract_xgrams(text: str, n_vals: typing.List[int]) -> typing.List[str]:
    """
    Extract a list of n-grams of different sizes from a text.
    Params:
        text: the test from which to extract ngrams
        n_vals: the sizes of n-grams to extract
        (e.g. [1, 2, 3] will produce uni-, bi- and tri-grams)
    """
    xgrams = []
    
    for n in n_vals:
        # if n > len(text) then no ngrams will fit, and we would return an empty list
        if n < len(text):
            for i in range(len(text) - n + 1) :
                ng = text[i:i+n]
                xgrams.append(ng)
        
    return xgrams


xgrams = extract_xgrams(text, n_vals=range(1,4))

print(xgrams)

def build_model(text: str, n_vals: typing.List[int]) -> typing.Dict[str, int]:
    """
    Build a simple model of probabilities of xgrams of various lengths in a text
    Parms:
        text: the text from which to extract the n_grams
        n_vals: a list of n_gram sizes to extract
    Returns:
        A dictionary of ngrams and their probabilities given the input text
    """
    model = collections.Counter(extract_xgrams(text, n_vals))  
    num_ngrams = sum(model.values())

    for ng in model:
        model[ng] = model[ng] / num_ngrams

    return model
  
  def calculate_cosine(a: typing.Dict[str, float], b: typing.Dict[str, float]) -> float:
    """
    Calculate the cosine between two numeric vectors
    Params:
        a, b: two dictionaries containing items and their corresponding numeric values
        (e.g. ngrams and their corresponding probabilities)
    """
    numerator = sum([a[k]*b[k] for k in a if k in b])
    denominator = (math.sqrt(sum([a[k]**2 for k in a])) * math.sqrt(sum([b[k]**2 for k in b])))
    return numerator / denominator

test_model = build_model(text, n_vals=range(1,4))
print({k: v for k, v in sorted(test_model.items(), key=lambda item: item[1], reverse=True)})

raw_texts = {language: udhr.raw(language_id) for language, language_id in zip(languages, language_ids)}
print(raw_texts['english'][:1000]) # Just print the first 1000 characters

# Build a model of each language
models = {language: build_model(text=raw_texts[language], n_vals=range(1,4)) for language in languages}
print(models['german'])

def identify_language(
    text: str,
    language_models: typing.Dict[str, typing.Dict[str, float]],
    n_vals: typing.List[int]
    ) -> str:
    """
    Given a text and a dictionary of language models, return the language model 
    whose ngram probabilities best match those of the test text
    Params:
        text: the text whose language we want to identify
        language_models: a Dict of Dicts, where each key is a language name and 
        each value is a dictionary of ngram: probability pairs
        n_vals: a list of n_gram sizes to extract to build a model of the test 
        text; ideally reflect the n_gram sizes used in 'language_models'
    """
    text_model = build_model(text, n_vals)
    language = ""
    max_c = 0
    for m in language_models:
        c = calculate_cosine(language_models[m], text_model)
        # The following line is just for demonstration, and can be deleted
        print(f'Language: {m}; similarity with test text: {c}')
        if c > max_c:
            max_c = c
            language = m
    return language

print(f"Test text: {text}")
print(f"Identified language: {identify_language(text, models, n_vals=range(1,4))}")

t = "mij werd geleerd dat de weg van vooruitgang noch snel noch gemakkelijk is."  
print(identify_language(t, models, n_vals=range(1,4)))

t = "on m'a appris que la voie du progrès n'était ni rapide ni facile."  
print(identify_language(t, models, n_vals=range(1,4)))

t = "me enseñaron que el camino hacia el progreso no es ni rápido ni fácil."
print(identify_language(t, models, n_vals=range(1,4)))
