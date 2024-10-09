import so
import re
import pandas as pd
import NumPy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
sns.set(font_scale=1.5)
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout, SpatialDropout1D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')
print('Training set has {} rows and {} columns'.format(train.shape[0], train.shape[1]))
print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
train.head()
train.isnull().sum() / train.shape[0] * 100
non_fake_len = train[train['target'] == 0]['text'].str.split().apply(lambda x: len(x))
fake_len = train[train['target'] == 1]['text'].str.split().apply(lambda x: len(x))

fig, ax = plt.subplots(1,2,figsize=(10,5))
sns.distplot(non_fake_len, ax=ax[0], kde=False, bins=15);
ax[0].set_xlabel('Not Fake');
sns.distplot(fake_len, ax=ax[1], kde=False, color='red', bins=15);
ax[1].set_xlabel('Fake');
fig.suptitle('Distribution of word length');
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
wordcloud = WordCloud(stopwords=set(list(STOPWORDS)), background_color='white')
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
wc0 = wordcloud.generate(str(train[train['target'] == 0]['text']))
ax[0].imshow(wc0);
ax[0].axis('off');
ax[0].set_title('Not Fake');
wc1 = wordcloud.generate(str(train[train['target'] == 1]['text']))
ax[1].imshow(wc1);
ax[1].axis('off');
ax[1].set_title('Fake');
fig.suptitle('Most Frequent Words');
X = train['text']
X_test = test['text']
y = train['target']
def clean_special_char(text):
    return re.sub('[^a-zA-Z0-9]', ' ', text)
def clean_lowercase(text):
    return str(text).lower()
stopwords = set(stopwords.words('english'))
def clean_stopwords(text):
    list_text = text.split()
    new_text = [word for word in list_text if word not in stopwords]
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
nltk.download('omw-1.4')
before_clean = X[:5]
X = X.apply(clean_special_char)
X = X.apply(clean_lowercase)
X = X.apply(clean_stopwords)
X = X.apply(clean_stem)
X = X.apply(clean_lem)
X_test = X_test.apply(clean_special_char)
X_test = X_test.apply(clean_lowercase)
X_test = X_test.apply(clean_stopwords)
X_test = X_test.apply(clean_stem)
X_test = X_test.apply(clean_lem)
for i in range(5):
    print(before_clean[i])
    print(X[i])
    print()
tokenizer = Tokenizer(num_words=3000)
tokenizer.fit_on_texts(X.values.tolist())
train_sequence = tokenizer.texts_to_sequences(X)
test_sequence = tokenizer.texts_to_sequences(X_test)
maxlen = 25
train_pad = pad_sequences(train_sequence, maxlen=maxlen)
test_pad = pad_sequences(test_sequence, maxlen=maxlen)
X_train, X_val, y_train, y_val = train_test_split(train_pad, y, test_size=0.3, random_state=123)
def create_model():
    embedding_dim = 16
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(1, activation='sigmoid'))
    return model
model = create_model()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=True, epochs=20, batch_size=32)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, 21)
plt.plot(epochs, acc);
plt.plot(epochs, val_acc);
plt.show()
plt.plot(epochs, loss);
plt.plot(epochs, val_loss);
plt.show()
pred = model.predict(test_pad)
result_df = pd.DataFrame(pred, columns=['target'])
result_df['id'] = test['id'].values
result_df                             
