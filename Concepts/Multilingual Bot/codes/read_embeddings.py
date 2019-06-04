import numpy as np
import pandas as pd

def convert_intents_into_numbers():
    data = pd.read_csv('text_48_intent.csv')
    intent = data['intent']
    unique_intent = set(intent)
    intent_dictionary = {}
    key = 0
    for single_intent in unique_intent:
        intent_dictionary[single_intent] = key
        key = key+1
    intents_in_numbers = []
    for single_intent in intent:
        intents_in_numbers.append(intent_dictionary[single_intent])
    return intents_in_numbers
intents = convert_intents_into_numbers()

def create_embed_csv(language):
    dim = 1024
    X = np.fromfile(('../raw_embedding/'+language+'.raw'), dtype=np.float32, count=-1)
    X.resize(X.shape[0]//dim, dim)
    X = pd.DataFrame(X)
    X['intent'] = intents    
    X.to_csv(('../embedding_csv/'+language+'.csv'), index_label='index')

languages = ['english', 'hindi', 'spanish', 'arabic']

for language in languages:
    create_embed_csv(language)