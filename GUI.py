
print("Loading ...")

import tkinter as tk
from tkinter import messagebox

from Preprocess import * 
from LSTMModel import * 

import torch 
import nltk
from nltk.corpus import stopwords

device = "cuda:0" if torch.cuda.is_available() else "cpu"
max_len = 1500
bg_color = "#7CCFCF"

print("UI loading ...")
print("Loading embedding matrix...")
glove_path = "./glove/glove.6B.50d.txt"
embedded_words = load_glove_embeddings(glove_path)
words_index = {word:index for index, word in enumerate(sorted(embedded_words.keys()))}
nltk.data.path.append("./stopwords")
stop_words = set(stopwords.words('english'))
embedding_matrix = create_embedding_matrix(embedded_words, words_index)

print("Loading model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTMModel(embedding_matrix = embedding_matrix, device = device)
model.load_state_dict(torch.load("./model/lstm.pth"))
model.eval()

print("Done")



# Function to analyze the sentiment (for demo purposes, it randomly decides positive or negative)
def analyze_sentiment():
    sentence = text_box.get("1.0", tk.END).strip()
    # Placeholder logic for sentiment analysis
    # Replace this with actual sentiment analysis logic
    if not sentence:
        messagebox.showwarning("Input Error", "Please enter some text.")
        result_label.config(text = "")
        return
    
    s = sentence_to_index_tokens(sentence, words_index, stop_words, max_len)
    prob = F.sigmoid(model(torch.tensor([s], device=device)))
    sentiment = "POSITIVE"
    if(prob < 0.5):
        sentiment = "NEGATIVE"
        prob = 1 - prob 
    result_label.config(text=f"{sentiment} : {prob*100:.1f}%", fg="green" if sentiment == "POSITIVE" else "red")

# Create the main window
root = tk.Tk()
root.title("Sentiment Analyzer")
root.geometry("600x500")
root.configure(bg=bg_color)

# Create a label for instructions
instruction_label = tk.Label(root, text="Enter text to analyze sentiment:", bg=bg_color, fg="black", font=("Arial", 22))
instruction_label.pack(pady=10)

# Create a text box for user input
text_box = tk.Text(root, height=10, width=50, font=("Arial", 12), wrap='word', padx=15, pady = 15)
text_box.pack(pady = 15)

# Create an Analyze button
analyze_button = tk.Button(root, text="Analyze", command=analyze_sentiment, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"), padx=10, pady=5)
analyze_button.pack(pady=10)

# Create a label to display the result
result_label = tk.Label(root, text="", bg=bg_color, font=("Arial", 14, "bold"))
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()