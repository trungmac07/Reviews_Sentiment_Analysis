from Preprocess import * 
from LSTMModel import * 

import nltk
from nltk.corpus import stopwords

import torch.nn.functional as F 

MAX_LEN = 15000

def main():
    print("Loading embedding matrix...")
    glove_path = "glove/glove.6B.50d.txt"
    embedded_words = load_glove_embeddings(glove_path)
    words_index = {word:index for index, word in enumerate(sorted(embedded_words.keys()))}
    nltk.data.path.append("./stopwords")
    stop_words = set(stopwords.words('english'))
    embedding_matrix = create_embedding_matrix(embedded_words, words_index)

    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMModel(embedding_matrix = embedding_matrix, device = device)
    model.load_state_dict(torch.load("./model/lstm-10.pth"))
    model.eval()

    print("Done")

    print("Input your sentence: ")

    input_sentence = input()

    s = sentence_to_index_tokens(input_sentence, words_index, stop_words)

    prob = F.sigmoid(model(torch.tensor([s], device=device)))

    if(prob >= 0.5):
        print("Positive: {:.2f}".format(prob.item()))
    else:
        print("Negative: {:.2f}".format(1 - prob.item()))

main()