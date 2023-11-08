#Importing necessary libraries

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
import numpy as np
#-----------------------------------------------------------------------

Lyrics_data1 = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Final_lyricsdata.csv')
Lyrics_data1['genres'].value_counts()

df = Lyrics_data1.copy()

#---------------------------------------------------------------------------------
df2 = Lyrics_data1.copy()

#Extracting only the necessary columns for feature extraction
df2 = df2[['genres', 'final_lyrics', 'id']]
#---------------------------------------------------------------------------------
#word2vec
#-----------------------------------------------------------------------------------

import pandas as pd
#word2vec
from gensim.models import Word2Vec
#To tekenize the text
from nltk.tokenize import word_tokenize
import numpy as np

#Hyperparameters for skip-gram
number_dimensions = 250
minimum_wordcount = 15
window_size = 5
seed_repro = 42
sg = 0  #  0 for CBOW

#Tokenizig the lyrics
df2["tokenized_words"] = df2["final_lyrics"].apply(word_tokenize)

#Trinaing the word2vec model
word2vec_model = Word2Vec(
    sentences=df2["tokenized_words"].tolist(),
    seeds =seed_repro,
    vector_size=number_dimensions,
    min_count=minimum_wordcount,
    window=window_size,
    sg=sg
)

#Vector representation of lyrics (each row)
vectors_eachrow = []
for tokenized_lyrics in df2["tokenized_words"]:
    vector_eachrow = np.mean([word2vec_model.wv[words] for words in tokenized_lyrics if words in word2vec_model.wv], axis=0)
    vectors_eachrow.append(vector_eachrow)

# Convert the list of row vectors to a DataFrame
CBOW = pd.DataFrame(vectors_eachrow, columns=[f"feature_{i}" for i in range(number_dimensions)])
# Reset the index for df2 and CBOW
df2.reset_index(drop=True, inplace=True)
CBOW.reset_index(drop=True, inplace=True)

#Adding id column to CBOW dataframe
CBOW['genres']= df2['genres']
CBOW['id'] = df2['id']

os.makedirs(r'C:\Dissertation\Music4all_dataset', exist_ok=True)  
CBOW.to_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\skip_gram.csv', index = False) 

#https://stackoverflow.com/questions/50373248/how-to-generate-word2vec-vectors-in-python
#https://builtin.com/machine-learning/nlp-word2vec-python
#-------------------------------------------------------------------------------------
#Glove
#-----------------------------------------------------------------------------------------


#Directory path containing glove embeddings
glove_directory = r"C:\Dissertation\Music4all_dataset\Vectorisation\glove.6B.100d.txt"

def embeddings_load():
    index_embeddings = {}
    with open(glove_directory, encoding="utf8") as f:
        for i in f:
            numbers = i.split()
            Words = numbers[0]
            coefficients = np.asarray(numbers[1:], dtype='float32')
            index_embeddings[Words] = coefficients
    return index_embeddings

#Maximum number of words to considered as features
max_num_words = 100000
#Dimensions 
emb_dimensions = 100

def embedding_matrix(indexword):
    index_embeddings = embeddings_load()
    num_words = min(max_num_words, len(indexword))
    matrix = np.zeros((num_words + 1, emb_dimensions))
    for word, i in indexword.items():
        if i >= max_num_words:
            continue
        vectors = index_embeddings.get(word)
        if vectors is not None:
            #if the word not present in the embeddings, then it will consider all as non-zeros
            matrix[i] = vectors
    return matrix, num_words

#Creating a new dataframe
lyrics_df = pd.DataFrame({"final_lyrics": df['final_lyrics']})

# Tokenize the lyrics into word sequences
def tokenization(lyrics):
    return lyrics.split()
df['tokenized_lyrics'] = df['final_lyrics'].apply(tokenization)

#splits the words, creates unique word and create ditionary, word index
Unique_words = df['tokenized_lyrics'].str.split().explode().unique()
Word_index = {words: i + 1 for i, words in enumerate(Unique_words)}
#https://stackoverflow.com/questions/38956274/how-to-find-index-of-an-exact-word-in-a-string-in-python

# Prepare embedding matrix and num_words
matrix, number_words = embedding_matrix(Word_index)

# Create a DataFrame with the GloVe embeddings for each lyrics text
def creating_emb_dataframe(lyrics_df):
    # Create an empty list to store the embeddings
    emb_list = []

    #Looping through each row
    for _, row in lyrics_df.iterrows():
        lyrics = row['tokenized_lyrics']
        emb_lists = [matrix[Word_index[word]] for word in lyrics if word in Word_index]
        #Calculating the average for all embeddings
        if emb_lists:
            average_emb = emb_lists.append(np.mean(emb_lists, axis=0))
        else:
            average_emb.append(np.zeros(emb_dimensions))

    #Creating a dataframe for obtained embeddings
    emb_columns = [f"embedding_{x}" for x in range(emb_dimensions)]
    emb_dataframe = pd.DataFrame(emb_lists, columns=emb_columns)

    return emb_dataframe

# Creating the final embedding DataFrame
embedding_dataframe = creating_emb_dataframe(df)
#Resetting the dataframes
df.reset_index(drop=True, inplace=True)
embedding_dataframe.reset_index(drop=True, inplace=True)
#Merging the id and genres with embedding vectors
lyrics_df = pd.concat([df['id'], embedding_dataframe], axis=1)
lyrics_df = pd.concat([df['genres'], lyrics_df], axis=1)

#Saving it in c drive
os.makedirs(r'C:\Dissertation\Music4all_dataset', exist_ok=True)  
lyrics_df.to_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Glove_basics.csv', index = False) 

#glove_basics no cleaning
#https://turbolab.in/text-classification-with-keras-and-glove-word-embeddings/

#------------------------------------------------------------------------------

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Load your data (replace with your data loading code)
data = pd.read_csv('your_data.csv')

df3 = df2.copy()
# Assuming you have lemmatized sentences in a column named 'lemmatized_sentences'
lemmatized_sentences = df['final_lyrics']

# Create TaggedDocument objects for Doc2Vec
tagged_data = [TaggedDocument(words=doc.split(), tags=[str(i)]) for i, doc in enumerate(lemmatized_sentences)]

# Initialize and train Doc2Vec model
model = Doc2Vec(vector_size=300, window=5, min_count=1, workers=4, epochs=20)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

# Create a DataFrame to store the document embeddings
document_embeddings = pd.DataFrame([model.dv[i] for i in range(len(tagged_data))])

# Add other columns from your original DataFrame if needed
document_embeddings = pd.concat([df3['id'], document_embeddings], axis=1)
document_embeddings = pd.concat([df3['genres'], document_embeddings], axis=1)

document_embeddings['genres'] = df3['genres']
document_embeddings['id'] = df3['id']
# Now, document_embeddings contains the vector representation of each record

# Continue with your analysis using the document embeddings
import os

os.makedirs(r'C:\Dissertation\Music4all_dataset', exist_ok=True)  
document_embeddings.to_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\document_embeddings.csv', index = False) 

#https://medium.com/@morga046/multi-class-text-classification-with-doc2vec-and-t-sne-a-full-tutorial-55eb24fc40d3#:~:text=In%20this%20article%2C%20I%20will%20show%20you%20how,this%20example%20I%20will%20be%20using%20these%20texts.
#------------------------------------------------------------------
