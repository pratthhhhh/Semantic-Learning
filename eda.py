import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

plt.style.use('fivethirtyeight')
sns.set(font_scale=1.5)

def run_eda(train_path='Dataset/train.csv', test_path='Dataset/test.csv'):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print('Training set has {} rows and {} columns'.format(train.shape[0], train.shape[1]))
    print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))
    print(train.head())
    
    missing_ratio = train.isnull().sum() / train.shape[0] * 100
    print("Missing ratio:\n", missing_ratio)
    
    non_fake_len = train[train['target'] == 0]['text'].dropna().str.split().apply(lambda x: len(x))
    fake_len = train[train['target'] == 1]['text'].dropna().str.split().apply(lambda x: len(x))
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    sns.distplot(non_fake_len, ax=ax[0], kde=False, bins=15)
    ax[0].set_xlabel('Not Fake')
    sns.distplot(fake_len, ax=ax[1], kde=False, color='red', bins=15)
    ax[1].set_xlabel('Fake')
    fig.suptitle('Distribution of word length')
    plt.show()
    
    wordcloud = WordCloud(stopwords=set(list(STOPWORDS)), background_color='white')
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    
    text0 = " ".join(train[train['target'] == 0]['text'].dropna().astype(str).tolist())
    if text0:
        wc0 = wordcloud.generate(text0)
        ax[0].imshow(wc0)
    ax[0].axis('off')
    ax[0].set_title('Not Fake')
    
    text1 = " ".join(train[train['target'] == 1]['text'].dropna().astype(str).tolist())
    if text1:
        wc1 = wordcloud.generate(text1)
        ax[1].imshow(wc1)
    ax[1].axis('off')
    ax[1].set_title('Fake')
    
    fig.suptitle('Most Frequent Words')
    plt.show()

if __name__ == '__main__':
    run_eda()
