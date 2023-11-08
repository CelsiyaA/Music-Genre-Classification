
#________________________________________________________________________________
#Enhancing music genre classification on fusing audio, lyrics
#______________________________________________________________________________

#------------------------------------------------------------------------------
#Importing the libraries
#------------------------------------------------------------------------------
#Data manipulation and analysis
import pandas as pd
#Numerical calculation
import numpy as np
#Natural language toolkit-deal with text data
import nltk              
#Regular expression - string searching and manipulation
import re
#For string manipulation
import string
string.punctuation
#Word_tokenize
from nltk.tokenize import word_tokenize
#Stop words corpus
from nltk.corpus import stopwords
#Lemmatizing
from nltk.stem import WordNetLemmatizer
#Textblob
from textblob import TextBlob
#Contarctions
import contractions
#To plot graphs and charts
import matplotlib.pyplot as plt
import seaborn as sns
#Regular expression
import re
#Directory
import os
#Countvectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
#Stop words corpus
from nltk.corpus import stopwords

#-----------------------------------------------------------------------------

#Reading the lyrics file
Lyrics = pd.read_csv("C:\Dissertation\Music4all_dataset\Lyrics.csv") #37353

#Check the duplicate records
print(len(Lyrics[Lyrics.duplicated(subset = 'lyrics', keep = "first")])) 

#Removing duplicate records
Lyrics.drop_duplicates(subset = "lyrics",keep = "first", inplace = True)

Lyrics['genres'].value_counts()

#----------------------------------------------------------------------------
#Reducing the size of data using proportionate stratified sampling
#---------------------------------------------------------------------------

def stratified_sampling(Lyrics, column_genre, fraction_of_samples):
    # Calculating sample size based on fraction of samples per genres (20%)
    sample_size = np.round(((Lyrics[column_genre].value_counts()) / len(Lyrics)) * len(Lyrics) * fraction_of_samples).astype(int)

    # selecting the sample size randomly based on sample fraction
    Random_sampleddata = pd.DataFrame()
    for genres, size in sample_size.items():
        Genredata = Lyrics[Lyrics[column_genre] == genres]
        Random_sampleddata = pd.concat([Random_sampleddata, Genredata.sample(n=size, random_state=42)])

    return Random_sampleddata

Sample_fraction = 0.2  
Reduced_lyrics = stratified_sampling(Lyrics, 'genres', Sample_fraction)

#https://forms.app/en/blog/stratified-random-sampling

#Saving the dataframe into c drive
os.makedirs(r'C:\Dissertation\Music4all_dataset', exist_ok=True)  
Reduced_lyrics.to_csv(r'C:\Dissertation\Music4all_dataset\Reduced_lyrics.csv', index = False)

#------------------------------------------------------------------------------
#Data preprocessing
#------------------------------------------------------------------------------

Reduced_lyrics = pd.read_csv("C:\Dissertation\Music4all_dataset\Reduced_lyrics.csv")

#Keeping the most popular genres
Genres = ['country','electronic','rap','rock','metal','jazz','soul','folk']
Data = Reduced_lyrics[Reduced_lyrics['genres'].isin(Genres)] #57 for blues

#Reading the data
Audio_features = pd.read_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Audio_features4000.csv',index_col = False)

#Merging the id of audio data with the lyrics data
df = pd.merge(Audio_features['id'], Reduced_lyrics, on = "id")

#Saving the dataframe into c drive
os.makedirs(r'C:\Dissertation\Music4all_dataset', exist_ok=True)  
df.to_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Lyrics_data.csv', index = False)

#------------------------------------------------------------------------------------

Lyrics_data = pd.read_csv("C:\Dissertation\Music4all_dataset\Vectorisation\Lyrics_data.csv")

#Shape of data
Lyrics_data.shape # 4531 records and 4 columns

#Information about the data
Lyrics_data.info()

#summarystatistics of genre
summarystatistics_genres = Lyrics_data['genres'].describe(include = 'object')

#Null values
Lyrics_data.isnull().sum() # no null values

#Checking the unique values
Lyrics_data.nunique() # this dataset contains only english language

df = Lyrics_data.copy()

#word count of lyrics data
def count_words(lyrics):
    #Split() function spilts the text into list of words
    words = lyrics.split()
    #returns the length of words
    return len(words)

# Creating a new column named word count to save the count of each row & applying a function
df['word_count'] = df['lyrics'].apply(count_words)


#https://www.formpl.us/blog/stratified-sampling
#https://www.questionpro.com/blog/stratified-random-sampling/#:~:text=In%20this%20approach%2C%20each%20stratum%20sample%20size%20is,hth%20stratum%20Nh%3D%20Population%20size%20for%20hth%20stratum

#---------------------------------------------------------------------------
#Exploratory data analysis
#----------------------------------------------------------------------------

#Histogram for word count of lyrics
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
plt.hist(df['word_count'], bins=20, edgecolor='k')
plt.xlabel('Values of word count')
plt.ylabel('Frequency occurence') #how many times a lyrics fall into word count range
plt.title('Distribution of word count in lyrics')
plt.show()
# most lyrics have value count range between 150 to 400

#-----------------------------------------------------------------------
#Pie chart - Distribution of genres
import matplotlib.pyplot as plt

#Genre names and its count of records
genre_names = ['rock', 'electronic', 'folk', 'soul', 'rap', 'country', 'metal', 'jazz']
count_genres = [1499, 758, 622, 604, 441, 260, 176, 171]
explode = (0.1, 0, 0, 0, 0, 0, 0, 0)
figure, axis = plt.subplots()
axis.pie(count_genres, explode=explode, labels=genre_names, autopct='%1.1f%%',
        shadow=True, startangle=90, pctdistance=0.85)
#Draw a circle
circle = plt.Circle((0, 0), 0.70, fc='white')
figure = plt.gcf()
figure.gca().add_artist(circle)

# Equal ratio for all genres
axis.axis('equal')
plt.tight_layout()
plt.title('Distribution of genres in lyrics')
plt.show()

#From the plot, we can seen that the genres are imbalanced.
#https://www.kdnuggets.com/2021/02/stunning-visualizations-using-python.html
#--------------------------------------------------------------------------------

#Repititive words
#Finding sequences of words in lyrics column
def top_ngram(lyrics, n=None):
    vectors = CountVectorizer(ngram_range=(n, n)).fit(lyrics)
    #Bag of words
    BOW = vectors.transform(lyrics)
    token_counts = BOW.sum(axis=0) 
    lyrics_words_freq = [(word, token_counts[0, idx]) 
                  for word, idx in vectors.vocabulary_.items()]
    lyrics_words_freq =sorted(lyrics_words_freq, key = lambda x: x[1], reverse=True)
    return lyrics_words_freq[:10]

#10 Top, 3 repetitive words of all lyrics
Tri_grams =top_ngram(df['lyrics'],n=3) #see also for n = 5
x_axis,y_axis =map(list,zip(*Tri_grams))
sns.barplot(x=y_axis,y=x_axis)

#10 Top, 2 repetitive words of all lyrics
Bigrams =top_ngram(df['lyrics'],2)[:10] 
x_axis,y_axis=map(list,zip(*Bigrams)) 
sns.barplot(x=y_axis,y=x_axis)

# Sub-plots for 3 to 8 grams
figure, axes = plt.subplots(nrows=6, ncols=1, figsize=(8,15))

for i, n in enumerate(list(range(3, 9))):
    # n_grams
    n_grams = top_ngram(df['lyrics'], n=n)
    x_axis, y_axis = map(list, zip(*n_grams))
    
    #Barplot for each grams
    sns.barplot(x=y_axis, y=x_axis, ax=axes[i])
    
    #Title, axis for plots
    axes[i].set_xlabel('Frequency count')
    axes[i].set_ylabel(f'{n}-grams')
    axes[i].set_title(f'Top {n} Repetitive {n}-grams in Lyrics')

# Adjust the layout
plt.tight_layout()
plt.title('Top 10 consecutive repetitive words of n-grams')
# Show the plot
plt.show()
#These repetative words will add noise to the data, so we remove it.

#https://towardsdatascience.com/a-complete-exploratory-data-analysis-and-visualization-for-text-data-29fb1b96fb6a
#---------------------------------------------------------------------

#word cloud
wordcloud = WordCloud(background_color="white").generate(Lyrics_data['lyrics'])
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

#---------------------------------------------

#Is digits present in the dataset
#Extracting digits from lyrics columns of each row
df['Digits'] = df['lyrics'].apply(lambda x: re.findall(r'\d+', x))

#Flattening the digits column
all_digits = [int(digits) for lists in df['Digits'] for digits in lists]

# Create a histogram to visualize the distribution of digits
plt.figure(figsize=(8, 6))
plt.hist(all_digits, bins=20, color='yellow', edgecolor='black')
plt.xlabel('Digits')
plt.ylabel('Frequency count')
plt.title('Histogram for digits in lyrics')
plt.show()

#________________________________________________________________________________________
# DATA-PREPROCESSING
#------------------------------------------------------------------------------------
#_____________________________________________________________________________________
#Data Explorarion - section 1
#Text preprocessing
         #Lower the words 
         #Expanding contractions
         #Remove the punctuations
         #Removing special characters
         #Repeated characters
         #Repeated words
         #Word tokenization
         #Remove the stopwords
         #Spelling correction #alter the indented meaning of the text
         #Removal of non-ASCII characters
         #Remove digits
         #Lemmatization
         #Stemming
         
#Second time - lower the words, punctuations, special characters, tokenization
#white space, stop words, lemmatization, la-la-la to be la la la, words notin eng dictionary
#________________________________________________________________________________________


lyrics_data = Lyrics_data.copy()

#Removing the unwanted column language
lyrics_data = lyrics_data.drop('lang', axis=1)

#Restting the index columns in a proper order
lyrics_data.reset_index(drop=True, inplace=True)
#-----------------------------------------------------------------------------
#Lowering the text

lyrics_data['lyrics'] = lyrics_data['lyrics'].apply(lambda x: x.lower())


#Expanding the contarctions
def expand_contractions(lyrics):
   expanded_lyrics = contractions.fix(lyrics)
   return expanded_lyrics

lyrics_data['lyrics'] = lyrics_data['lyrics'].apply(expand_contractions)

# Removal of punctuation

def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])

lyrics_data['lyrics'] = lyrics_data['lyrics'].apply(lambda x: remove_punctuation(x))

#Removing special characters

lyrics_data['lyrics'] = lyrics_data['lyrics'].replace(r'[^A-Za-z0-9 ]+', '', regex=True)

#Removing repeated characters
def repeated_characters(lyrics):
    # findind words with characters repeated more than twice
     Finding_repeated_char = re.compile(r'(\w)\1{2,}')
    # replacing the repetition of character with single occurence
     Replacing_1_occ = Finding_repeated_char.sub(r'\1', lyrics)
     return  Replacing_1_occ

lyrics_data['spec_character_removal'] = lyrics_data['lyrics'].apply(repeated_characters)

#https://stackoverflow.com/questions/69465843/remove-repeating-characters-from-sentence-but-retain-the-words-meaning


from collections import Counter
#gpt
def repeatedwords_removal(text):
    #spilting the text data
    words = text.split()
    
    #list to store the words
    new_words = []
    #Looping through the word
    for word in words:
        #It will add words to new_words list, if the current word is not same as the previous word
        if not new_words or word != new_words[-1]:
            new_words.append(word)
    #Joining the new words into a string
    removed_words = ' '.join(new_words)
    
    return removed_words

# Apply the remove_repetitive_words function to each row in the DataFrame
lyrics_data['repeatedwords_removal'] = lyrics_data['spec_character_removal'].apply(repeatedwords_removal)

#https://datascience.stackexchange.com/questions/34039/regex-to-remove-repeating-words-in-a-sentence

def remove_certain_words(lyrics):
    remove_words = ['chorus', 'verses', 'verse', 'intro', 'bridge', 'prechorus', 'pre']
    splitting = lyrics.split()
    return ' '.join([words for words in splitting if words not in remove_words])

lyrics_data['remove_words'] = lyrics_data['repeatedwords_removal'].apply(remove_certain_words)

#https://stackoverflow.com/questions/29771168/how-to-remove-words-from-a-list-in-python

#Removal of digits
lyrics_data['digits'] = lyrics_data['remove_words'].str.replace(r'\d+','',regex = True)

#Word_tokenization - splitting the sentence into words or tokens

lyrics_data['tokenization']= lyrics_data['digits'].apply(lambda X: word_tokenize(X))

#Removal of stopwords- Stop words are words which cannot give much meaning to the sentence

def remove_stopwords(lyrics):
    result = []
    for tokens in lyrics:
        if tokens not in stopwords.words('english'):
            result.append(tokens)
    return result

lyrics_data['stop_words'] = lyrics_data['tokenization'].apply(remove_stopwords)

#Non-lexical words

# Words to be removed
words_to_remove = ["ah", "ahh", "aahaah", "aahaahaah","aahh","aahhaa","aahho","oh","whoa","ooo"
                   "hum","ahhahh","ahoh","ooh", "yeah","oohooh","la","ooooh","hey","woah","na","wo",
                   "aaa","aa","aayaaaa","yaa","haa","mmm","mm","mmmh","oooo","ooo","yae","ly","eeeyyyeeaaaah"
                   "ummmm","ohhhh","ha","uh","nanana","ahah","mhmm","mmmm","tu","ch","nah","ee",
                   "ay","ba","yay","buhbuh","aaaahhh","aaaaahh","aaaaaaah","bobobobobobobo","bo",
                   "ai","yai","ayy","dodo","ahhyeeah","pew","aye","De","boom","ou","ya","yay","ooww","lalala",
                   "uhhuh","woahohoh","mmmmm","nananana","lala","ba","da","ooh","ohh","lala","la","na"]

# Function to remove specified words from tokenized data
def remove_words(tokens):
    return [word for word in tokens if word not in words_to_remove]

# Remove specified words from the DataFrame
lyrics_data['non-lexical_vocals'] = lyrics_data['stop_words'].apply(remove_words)

#Spelling correction

def spelling_correction(lyrics):
    #Correcting the words and joining it in single string
    text = ' '.join([str(TextBlob(word).correct()) for word in lyrics])
    
    return text

# Apply the spelling_correction function to your DataFrame column
lyrics_data['corrected_lyrics'] = lyrics_data['non-lexical_vocals'].apply(spelling_correction)


#Removing non-ASCII characters
def non_ASCII(lyrics):
    #string without non-ASCII characters
    return ''.join(i for i in lyrics if 0 < ord(i) < 127)

lyrics_data['corrected_lyrics'] = lyrics_data['corrected_lyrics'].apply(non_ASCII)

#https://stackoverflow.com/questions/2743070/remove-non-ascii-characters-from-a-string-using-python-django

def remove_repetitive_words(text):
    # Split the text into words
    words = text.split()
    # Initialize a list to store unique words
    unique_words = []
    # Iterate through the words
    for word in words:
        # If the current word is not the same as the previous word, add it to the unique_words list
        if not unique_words or word != unique_words[-1]:
            unique_words.append(word)
    
    # Join the unique words back into a single string
    cleaned_text = ' '.join(unique_words)
    return cleaned_text

# Apply the remove_repetitive_words function to each row in the DataFrame
lyrics_data['repeatedwords_removal2'] = lyrics_data['corrected_lyrics'].apply(remove_repetitive_words)


#Lemmitization

def lemma(lyrics):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in lyrics.split()]
    return ' '.join(lemmatized_tokens)

lyrics_data['lemmatized_lyrics'] = lyrics_data['repeatedwords_removal2'].apply(lemma)
#Removing extra spaces

lyrics_data['final_lyrics'] = lyrics_data['lemmatized_lyrics'].apply(lambda x: re.sub(' +', ' ', x))

lyrics_data['final_lyrics'] = lyrics_data['final_lyrics'].apply(lambda x: re.sub(' +', ' ', x)) 

import os
os.makedirs(r'C:\Dissertation\Music4all_dataset', exist_ok=True)  
lyrics_data.to_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\Final_lyricsdata.csv', index = False) 

#---------------------------------------------------------------------------
#II - Text preprocessing
#--------------------------------------------------------------------------

lyrics_data2 = lyrics_data.copy()

#Removing the unwanted column language
lyrics_data2 = lyrics_data2.drop('lang', axis=1)

#Restting the index columns in a proper order
lyrics_data2.reset_index(drop=True, inplace=True)

#Lowering the text

lyrics_data2['lyrics'] = lyrics_data2['lyrics'].apply(lambda x: x.lower())

#Expanding the contarctions
def expand_contractions(lyrics):
   expanded_lyrics = contractions.fix(lyrics)
   return expanded_lyrics

lyrics_data2['lyrics'] = lyrics_data2['lyrics'].apply(expand_contractions)

# Removal of punctuation

def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])

lyrics_data2['lyrics'] = lyrics_data2['lyrics'].apply(lambda x: remove_punctuation(x))

#Removing special characters

lyrics_data2['lyrics'] = lyrics_data2['lyrics'].replace(r'[^A-Za-z0-9 ]+', '', regex=True)

#Removal of words
def remove_certain_words(lyrics):
    remove_words = ['chorus', 'verses', 'verse', 'intro', 'bridge', 'prechorus']
    splitting = lyrics.split()
    return ' '.join([words for words in splitting if words not in remove_words])

lyrics_data2['remove_words'] = lyrics_data2['lyrics'].apply(remove_certain_words)

#Word_tokenization - splitting the sentence into words or tokens

lyrics_data2['tokenization']= lyrics_data2['remove_words'].apply(lambda X: word_tokenize(X))

#Removal of stopwords- Stop words are words which cannot give much meaning to the sentence

def remove_stopwords(lyrics):
    result = []
    for tokens in lyrics:
        if tokens not in stopwords.words('english'):
            result.append(tokens)
    return result

lyrics_data2['stop_words'] = lyrics_data2['tokenization'].apply(remove_stopwords)

#Chatgpt
def spelling_correction(lyrics):
    # Correct each word individually using a list comprehension
    corrected_words = [str(TextBlob(word).correct()) for word in lyrics]
    
    # Join the corrected words back into a single string
    corrected_text = ' '.join(corrected_words)
    
    return corrected_text

# Apply the spelling_correction function to your DataFrame column
lyrics_data2['corrected_lyrics'] = lyrics_data2['stop_words'].apply(spelling_correction)

#Lemmitization

def lemma(lyrics):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in lyrics.split()]
    return ' '.join(lemmatized_tokens)

lyrics_data2['lemma'] = lyrics_data2['corrected_lyrics'].apply(lemma)

#Removing extra spaces

lyrics_data2['final_lyrics'] = lyrics_data2['lemma'].apply(lambda x: re.sub(' +', ' ', x))

os.makedirs(r'C:\Dissertation\Music4all_dataset', exist_ok=True)  
lyrics_data2.to_csv(r'C:\Dissertation\Music4all_dataset\Vectorisation\lyrics_basics.csv', index = False) 

#-----------------------------------------------------------------------------------------
