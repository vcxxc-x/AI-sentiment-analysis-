
print("""
      
    ██████ ▓█████  ███▄    █ ▄▄▄█████▓ ██▓ ███▄ ▄███▓▓█████  ███▄    █ ▄▄▄█████▓    ▄▄▄       ███▄    █  ▄▄▄       ██▓   ▓██   ██▓▒███████▒▓█████  ██▀███  
▒██    ▒ ▓█   ▀  ██ ▀█   █ ▓  ██▒ ▓▒▓██▒▓██▒▀█▀ ██▒▓█   ▀  ██ ▀█   █ ▓  ██▒ ▓▒   ▒████▄     ██ ▀█   █ ▒████▄    ▓██▒    ▒██  ██▒▒ ▒ ▒ ▄▀░▓█   ▀ ▓██ ▒ ██▒
░ ▓██▄   ▒███   ▓██  ▀█ ██▒▒ ▓██░ ▒░▒██▒▓██    ▓██░▒███   ▓██  ▀█ ██▒▒ ▓██░ ▒░   ▒██  ▀█▄  ▓██  ▀█ ██▒▒██  ▀█▄  ▒██░     ▒██ ██░░ ▒ ▄▀▒░ ▒███   ▓██ ░▄█ ▒
  ▒   ██▒▒▓█  ▄ ▓██▒  ▐▌██▒░ ▓██▓ ░ ░██░▒██    ▒██ ▒▓█  ▄ ▓██▒  ▐▌██▒░ ▓██▓ ░    ░██▄▄▄▄██ ▓██▒  ▐▌██▒░██▄▄▄▄██ ▒██░     ░ ▐██▓░  ▄▀▒   ░▒▓█  ▄ ▒██▀▀█▄  
▒██████▒▒░▒████▒▒██░   ▓██░  ▒██▒ ░ ░██░▒██▒   ░██▒░▒████▒▒██░   ▓██░  ▒██▒ ░     ▓█   ▓██▒▒██░   ▓██░ ▓█   ▓██▒░██████▒ ░ ██▒▓░▒███████▒░▒████▒░██▓ ▒██▒
▒ ▒▓▒ ▒ ░░░ ▒░ ░░ ▒░   ▒ ▒   ▒ ░░   ░▓  ░ ▒░   ░  ░░░ ▒░ ░░ ▒░   ▒ ▒   ▒ ░░       ▒▒   ▓▒█░░ ▒░   ▒ ▒  ▒▒   ▓▒█░░ ▒░▓  ░  ██▒▒▒ ░▒▒ ▓░▒░▒░░ ▒░ ░░ ▒▓ ░▒▓░
░ ░▒  ░ ░ ░ ░  ░░ ░░   ░ ▒░    ░     ▒ ░░  ░      ░ ░ ░  ░░ ░░   ░ ▒░    ░         ▒   ▒▒ ░░ ░░   ░ ▒░  ▒   ▒▒ ░░ ░ ▒  ░▓██ ░▒░ ░░▒ ▒ ░ ▒ ░ ░  ░  ░▒ ░ ▒░
░  ░  ░     ░      ░   ░ ░   ░       ▒ ░░      ░      ░      ░   ░ ░   ░           ░   ▒      ░   ░ ░   ░   ▒     ░ ░   ▒ ▒ ░░  ░ ░ ░ ░ ░   ░     ░░   ░ 
      ░     ░  ░         ░           ░         ░      ░  ░         ░                   ░  ░         ░       ░  ░    ░  ░░ ░       ░ ░       ░  ░   ░     
                                                                                                                        ░ ░     ░                        """)

import sys
import subprocess
import re

def install_libraries():
    packages = ["tensorflow", "pandas", "numpy", "scikit-learn"]
    subprocess.run([sys.executable, "-m", "pip", "install"] + packages)

# Uncomment to Quick Install Libraries 
# install_libraries()

#=========================================================================
#                        Main Imports 
#=========================================================================

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

model = None
tokenizer = None

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)  # remove URLs, mentions, hashtags
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text)  # remove extra whitespace
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # remove emojis and non-ASCII
    return text.strip().lower()

def load_csv_data():
    path = input("Enter path to CSV (must have 'text' and 'sentiment' columns): ")
    try:
        df = pd.read_csv(path)
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            print("CSV must contain 'text' and 'sentiment' columns.")
            return None
        df['text'] = df['text'].astype(str).apply(clean_text)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def preprocess_data(df):
    global tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    return padded, np.array(df['sentiment'])

def build_cnn_model():
    model = Sequential([
        Embedding(5000, 64, input_length=100),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def build_rnn_model():
    model = Sequential([
        Embedding(5000, 64, input_length=100),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=5, batch_size=2, verbose=1)
    predictions = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

def predict_sentiment(model, tokenizer):
    text = input("Enter text to analyze: ")
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = round(float(prediction), 3)
    judgment = "Highly Confident" if confidence > 0.85 or confidence < 0.15 else "Uncertain"
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence}")
    print(f"Judgment: {judgment}")
    
#====================================================================================
                                     #main
#====================================================================================

def main():
    global model, tokenizer
    print("""
==============================================================================
           Sentiment Analyzer Menu
==============================================================================   
                 
1> Import CSV & Train NLP
2> Predict and Judge
3> Exclude Comments {slur}
4> Re-Install Libraries
5> README
6> Exit 
    
""")
    while True:
        choice = input("Choose an option: ")
        if choice == '1':
            df = load_csv_data()
            if df is not None:
                padded, labels = preprocess_data(df)
                X_train, X_test, y_train, y_test = train_test_split(padded, labels, test_size=0.2)
                print("Choose model type:\n 1. CNN\n 2. RNN")
                model_choice = input("Model: ")
                if model_choice == '1':
                    model = build_cnn_model()
                elif model_choice == '2':
                    model = build_rnn_model()
                else:
                    print("Invalid model choice.")
                    continue
                train_model(model, X_train, y_train, X_test, y_test)
        elif choice == '2':
            if model and tokenizer:
                predict_sentiment(model, tokenizer)
            else:
                print("Please train a model first.")
        elif choice == '3':
            df = load_csv_data()
            if df is not None:
                slur_patterns = [r"\bidiot\b", r"\bstupid\b", r"\bfool\b", r"\bhate\b", r"\btrash\b"]
                pattern = re.compile("|".join(slur_patterns), re.IGNORECASE)
                filtered_df = df[~df['text'].str.contains(pattern)]
                filtered_df.to_csv("cleaned_comments.csv", index=False)
                print("Filtered comments saved to 'cleaned_comments.csv'")
        elif choice == '4':
            install_libraries()
        elif choice == "5":
            print("README: Select 1 and import .csv then train model, thereafter you can use options 2 & 3" \
            "Exclude has been integrated to silence off slurs")    
        elif choice == '6':
            print("Sentiment Analyzer, signing off... Until next time ;))")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
