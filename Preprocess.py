import numpy as np
import torch 

import re

def load_glove_embeddings(file_path):
    embedded_words = {}
    embedded_words[' '] = np.array([0] * 50, dtype='float32')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedded_words[word] = vector

    
    return embedded_words


def sentence_to_index_tokens(sentence, words_index, stop_words, max_len = 10000):
    sentence = sentence.lower()
    sentence = re.sub(r'[\']', ' ', sentence)
    sentence = re.sub(r'<[^>]{0,5}>|[^a-zA-Z\s]', ' ', sentence)
    tokens = sentence.split()
    tokens = [words_index[token] for token in tokens if token not in stop_words and token in words_index]
    if(len(tokens) < max_len):
        tokens += [0] * (max_len - len(tokens))
    return tokens[:max_len]

def preprocess_training_data(sentences, labels, words_index, stop_words, max_len = 10000):
    preprocessed_data = []

    for i,(sentence,label) in enumerate(zip(sentences,labels)):
        tokens = sentence_to_index_tokens(sentence, words_index, stop_words,  max_len=max_len)
        preprocessed_data.append((torch.tensor(tokens),torch.tensor(label)))
    
    return preprocessed_data


def create_embedding_matrix(embedded_words, words_index):
    embedding_matrix = np.zeros((len(words_index), 50), dtype='float32')
    for word, i in words_index.items():
        embedding_vector = embedded_words.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        
    
    return torch.tensor(embedding_matrix)
    
