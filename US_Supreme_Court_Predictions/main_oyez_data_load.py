import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV

import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import RegexpTokenizer


# !pip3 uninstall keras-nightly
# !pip3 uninstall -y tensorflow
# !pip3 install keras==2.1.6
# !pip3 install tensorflow==2.16.1
# !pip3 install h5py==2.10.0


import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import seaborn as sns
import pydot


# For displaying facts
pd.set_option('display.max_colwidth', None)

# Load dataset as dataframe
df = pd.read_pickle('https://github.com/yesol-ba/portfolio/blob/main/Data/ba865_supreme%20court%20data_task1_data.pkl?raw=true') #publicly accessible data path

#df = pd.read_pickle('/content/task1_data.pkl') # Sally's Path
df.rename(columns={'Facts': 'facts'}, inplace=True)
df.drop(columns=['index'], inplace=True)
df.reset_index(inplace=True)

print(f'There are {len(df)} cases.')


# Looking at the dataset
df.head(1)

# There are 3 numerical columns and 6 object columns
df.info()


# There isn't any missing values in this dataset
df.isna().sum()


avg_char = df['facts'].apply(lambda x: len(str(x))).mean()
print(f'Average facts character length: {avg_char:.0f}')

avg_word = df['facts'].apply(lambda x: len(str(x).split())).mean()
print(f'Average facts word length: {avg_word:.0f}')

del avg_char, avg_word


print(f'There are {len(df)} cases.')
print(f'There are {len(df[df["winner_index"]==0])} rows for class 0.')
print(f'There are {len(df[df["winner_index"]==1])} rows for class 1.')

# Facts character stats
df['facts'].apply(lambda x: len(str(x))).describe()

# Facts word stats
df['facts'].apply(lambda x: len(str(x).split())).describe()



# Seqence Model Check (Not Pass)
text_vectorization = keras.layers.TextVectorization(
    max_tokens=1000, # adding more tokens to allow for increase due to bigrams.
    output_mode="multi_hot", # This is requesting integer encodings (which means we'll have a sequence of integers)
)
text_vectorization.adapt(df['facts'])
vectorized_facts = text_vectorization(df['facts'])


lengths = [len(x) for x in vectorized_facts]

print(f'The average fact in our data has {np.mean(lengths):.0f} words, and we have {len(df)} samples.\n')

print(f'The ratio of samples to average sample length is {(len(df)/np.mean(lengths)):.0f}. We are nowhere close to 1500.\n')

print(f'We need a larger dataset containing at least {(np.mean(lengths)*1500):.0f} samples.')



name_pet = []
name_rep = []
for i in range(df.shape[0]):
  fact = df["facts"][i]
  petitioner = df["first_party"][i]
  respondent = df["second_party"][i]
  p = True
  r = True
  for _ in petitioner.split():
    if _ in fact:
      p = True
      break
    else:
      p = False
  if p == False: 
    #name_pet.append("Petitioner name not found in {}".format(i))
    name_pet.append(i)
  for _ in respondent.split():
    if _ in fact:
      r = True
      break
    else:
      r = False
  if r == False:
    #name_rep.append("Respondent name not found in {}".format(i))
    name_rep.append(i)
    
    
perc_miss_pet = len(name_pet) / len(df) * 100
print('{:.2f}% of facts don\'t contain the first party name'.format(perc_miss_pet))

perc_miss_rep = len(name_rep) / len(df) * 100
print('{:.2f}% of facts don\'t contain the second party name'.format(perc_miss_rep))

perc_miss_both = len(set(set(name_pet) & set(name_rep))) / len(df) * 100
print('{:.2f}% of facts don\'t contain both first party the second party names'.format(perc_miss_both))


# Combining first party and second party with facts
df['facts'] = df['first_party']+' '+df['second_party']+' '+df['facts']


df['facts'][50]

print(df["winner_index"].value_counts())

df.groupby('winner_index').size().plot(kind='pie',
                                       y = "winner_index",
                                       label = "Type",
                                       autopct='%1.1f%%')


# %%

# Perform an 80-20 split for training and testing data
X_train, X_test, \
y_train, y_test = train_test_split(
    df[['winner_index', 'facts']],
    df['winner_index'],
    test_size=0.2,
    stratify=df['winner_index'],
    random_state=865
)

petitioner = X_train[X_train["winner_index"] == 0]
respondent = X_train[X_train["winner_index"] == 1]
print(petitioner.shape)
print(respondent.shape)

#upsampling respondents to match petioners

from sklearn.utils import resample
upsample_respondent = resample(respondent,
             replace=True,
             n_samples=len(petitioner),
             random_state=865)

upsample_train = pd.concat([upsample_respondent, petitioner])

print(upsample_train["winner_index"].value_counts())

upsample_train.groupby('winner_index').size().plot(kind='pie',
                                       y = "winner_index",
                                       label = "Type",
                                       autopct='%1.1f%%')


# Let's shuffle things... 
shuffled_indices= np.arange(upsample_train.shape[0])
np.random.shuffle(shuffled_indices)


shuffled_train = upsample_train.iloc[shuffled_indices,:]

X_train= shuffled_train['facts']

y_train = shuffled_train['winner_index']

# Dropping winner_index in X_test set
X_test = X_test['facts']


# define tokenizer function

def nltk_tokenizer(_wd):
  return RegexpTokenizer(r'\w+').tokenize(_wd.lower())


# turn X data into pandas dataframe to use custom defined function on a column in dataframe
X_train_frame = X_train.to_frame()
X_test_frame = X_test.to_frame()

# tokenize facts
X_train_frame["tokenized_facts"] = X_train_frame["facts"].apply(nltk_tokenizer)
X_test_frame["tokenized_facts"] = X_test_frame["facts"].apply(nltk_tokenizer)

# make taggeddocument, which is required format to use Doc2vec
tokens_train = X_train_frame["tokenized_facts"].to_list()
docs_train = [TaggedDocument(t, [str(i)]) for i, t in enumerate(tokens_train)]
tokens_test = X_test_frame["tokenized_facts"].to_list()
docs_test = [TaggedDocument(t, [str(i)]) for i, t in enumerate(tokens_test)]


# innitiate doc2vec model and train it on train data
doc2vec_model = Doc2Vec(vector_size=50, min_count=2, epochs=40, dm=1, seed=865, window=5)
doc2vec_model.build_vocab(docs_train)
doc2vec_model.train(docs_train, total_examples = doc2vec_model.corpus_count, epochs = doc2vec_model.epochs)


# vectorize train and test data using doc2vec model
X_train_dvs = [doc2vec_model.infer_vector(doc) for doc in tokens_train]
X_train_dvs = pd.DataFrame(X_train_dvs, index = X_train.index)

X_test_dvs = [doc2vec_model.infer_vector(doc) for doc in tokens_test]
X_test_dvs = pd.DataFrame(X_test_dvs, index = X_test.index)

