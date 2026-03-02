import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import pickle

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stopwords_set = set(stopwords.words('english'))

def clean_special_char(text):
    return re.sub('[^a-zA-Z0-9]', ' ', str(text))

def clean_lowercase(text):
    return str(text).lower()

def clean_stopwords(text):
    list_text = text.split()
    new_text = [word for word in list_text if word not in stopwords_set]
    return ' '.join(new_text)

def clean_stem(text):
    stemmer = PorterStemmer()
    list_text = text.split()
    new_text = [stemmer.stem(word) for word in list_text]
    return ' '.join(new_text)

def clean_lem(text):
    lemma = WordNetLemmatizer()
    list_text = text.split()
    new_text = [lemma.lemmatize(word=word, pos='v') for word in list_text]
    return ' '.join(new_text)

def clean_text(X):
    X = X.apply(clean_special_char)
    X = X.apply(clean_lowercase)
    X = X.apply(clean_stopwords)
    X = X.apply(clean_stem)
    X = X.apply(clean_lem)
    return X

def preprocess_data(train_path='Dataset/train.csv', test_path='Dataset/test.csv', num_words=3000, maxlen=25):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X = train['text']
    X_test = test['text']
    y = train['target']
    
    print("Cleaning training data...")
    X = clean_text(X)
    
    print("Cleaning test data...")
    X_test = clean_text(X_test)
    
    print("Tokenizing data...")
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X.values.tolist())
    
    train_sequence = tokenizer.texts_to_sequences(X)
    test_sequence = tokenizer.texts_to_sequences(X_test)
    
    train_pad = pad_sequences(train_sequence, maxlen=maxlen)
    test_pad = pad_sequences(test_sequence, maxlen=maxlen)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        
    return train_pad, y, test_pad, vocab_size, test

if __name__ == '__main__':
    train_pad, y, test_pad, vocab_size, test = preprocess_data()
    print(f"Preprocessing finished. Train pad shape: {train_pad.shape}, Vocab size: {vocab_size}")
