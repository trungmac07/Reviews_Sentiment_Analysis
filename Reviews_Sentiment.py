import os
import argparse

from Preprocess import * 
from LSTMModel import * 

import nltk
from nltk.corpus import stopwords

import torch.nn.functional as F 

from torch.utils.data import random_split, DataLoader, Dataset

import pandas as pd



def main(config):
   
    max_len = config.max_len

    if(config.device == "auto"):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: 
        device = config.device

    if(config.mode == "test"):
        print("Test mode")
        print("Loading embedding matrix...")
        glove_path = config.embedding_path
        embedded_words = load_glove_embeddings(glove_path)
        words_index = {word:index for index, word in enumerate(sorted(embedded_words.keys()))}
        nltk.data.path.append(config.stopwords_path)
        stop_words = set(stopwords.words('english'))
        embedding_matrix = create_embedding_matrix(embedded_words, words_index)

        print("Loading model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = LSTMModel(embedding_matrix = embedding_matrix, device = device)
        model.load_state_dict(torch.load(config.model_path))
        model.eval()

        print("Done")

        print("Input your sentence: ")

        test_path = config.test_path

        i = 1
        for path in test_path:
            f = open(path, 'r')
            for sentence in f:
                s = sentence_to_index_tokens(sentence, words_index, stop_words, max_len)
                prob = F.sigmoid(model(torch.tensor([s], device=device)))
                print(f"{i} : {sentence}")
                if(prob >= 0.5):
                    print("Positive: {:.2f}".format(prob.item()))
                else:
                    print("Negative: {:.2f}".format(1 - prob.item()))
                print("--------------------------------------------------------------------")
                i += 1

    elif(config.mode == "train"):
        print("Train mode")

        print("Loading embedding matrix...")
        glove_path = config.embedding_path
        embedded_words = load_glove_embeddings(glove_path)
        words_index = {word:index for index, word in enumerate(sorted(embedded_words.keys()))}
        nltk.data.path.append(config.stopwords_path)
        stop_words = set(stopwords.words('english'))
        embedding_matrix = create_embedding_matrix(embedded_words, words_index)

        print("Loading dataset")
        df = pd.read_csv(config.data_dir)
        df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
        sentences = df['review'].tolist()
        labels = df['label'].tolist()

        data = preprocess_training_data(sentences, labels, words_index, stop_words, max_len=max_len)
        train_len = int(0.8 * len(data))
        test_len = len(data) - train_len
        BATCH_SIZE = config.batch_size
        train, test = random_split(data, [train_len, test_len], generator=torch.Generator().manual_seed(77))
        train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test, batch_size=BATCH_SIZE)

        print(f"Loading model on step {config.start_step}")
        model = LSTMModel(embedding_matrix = embedding_matrix, device = device)
        
        if(config.start_step > 0):
            model.load_state_dict(torch.load(config.model_path))

        print(f"Training model on step {config.start_step}")
        
        fit(model, train_loader, config, device=device)

        print(f"Evaluating model on step {config.num_steps}")
        accuracy = eval(model, test_loader, config, device=device)

        print(f"Accuracy:",accuracy)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--max_len', type=int, default=1500, help='maximum lenght of sentences')
    parser.add_argument('--embedding_path', type=str, default="./glove/glove.6B.50d.txt", help="path to text embedding file")
    parser.add_argument('--stopwords_path', type=str, default="./stopwords", help="path to stopwords file")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='device (cpu or cuda)')

    
    # Training configuration.
    parser.add_argument('--num_steps', type=int, default=10, help='number of total steps for training')
    parser.add_argument('--start_step', type=int, default=0, help='resume training from this step')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')

    
    # Miscellaneous.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    
    # Directories.
    parser.add_argument('--data_dir', type=str, default="./data/IMDB Dataset.csv", help='data directory')
    parser.add_argument('--model_save_dir', type=str, default='./model', help = 'model directory for loading and saving in training')
    parser.add_argument('--test_path', nargs='+', default=[], help='Test file paths for testing. Each line will be a review sentence for analysis')
    parser.add_argument('--model_path', type=str, default='./model/lstm.pth', help='model path to load')

    # Step size.
    parser.add_argument('--model_save_step', type=int, default=5)

   

    config = parser.parse_args()
    
    main(config)


# TEST  : python.exe Reviews_Sentiment.py --mode test --model_path ./model/lstm.pth --test_path ./test/test.txt 
# TRAIN : python.exe Reviews_Sentiment.py --mode train --model_path ./model/lstm.pth --embedding_path ./glove/glove.6B.50d.txt --stopwords_path ./stopwords --data_dir "./data/IMDB Dataset.csv" --start_step 0 --num_steps 100 --batch_size 32 --model_save_step 5