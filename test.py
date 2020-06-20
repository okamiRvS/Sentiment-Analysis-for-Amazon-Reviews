import json
import pdb

import re
import string
from afinn import Afinn
import numpy as np
import pandas as pd
from collections import Counter
import itertools
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize

import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud

from afinn import Afinn
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation as LDA
import seaborn as sns

from pyLDAvis import sklearn as sklearn_lda
import pickle 
import pyLDAvis
import csv

# Function to get the counter
def get_counter(df):
	# Let's count the frequency of each word in the dataset.
	sentences = (list(itertools.chain(df)))
	flat_list = [item for sublist in sentences for item in sublist]

	# Let's print the most common tokens.
	c = Counter(flat_list)
	return c

def visualizeWords(df):
	sentences = (list(itertools.chain(df)))
	flat_list = [item for sublist in sentences for item in sublist]

	fig = plt.figure(figsize=(20,14))
	wordcloud = WordCloud(background_color="white").generate(" ".join(flat_list))
	plt.imshow(wordcloud,interpolation='bilinear')
	plt.axis("off")
	plt.show()

# Set pandas to see all data
def viewAllDataPandas():
	pd.set_option('display.max_columns', None)
	pd.set_option('display.max_rows', None)
	pd.set_option('display.width', None)
	pd.set_option('display.max_colwidth', -1)
	pd.set_option('display.max_rows', None)
	pd.set_option('display.max_columns', None)

def addStopwords():
	stop = stopwords.words("english") + stopwords.words("italian")
	stop = set(stop)
	stop.add("é")
	stop.add("ed")
	stop.add("là")
	stop.add("d1")
	stop.add("b")
	stop.add("a")
	stop.add("it")
	stop.add("the")
	stop.add("and")
	stop.add("http")
	stop.add("https")
	stop.add("co")
	stop.add("ce")
	stop.add("qn")
	stop.add("mj")
	stop.add("gb")
	stop.add("già")
	stop.add("ló")
	stop.add("ìn")
	stop.add("à")
	stop.add("ló")
	stop.add("ló")
	return list(stop)

def addStranges():
	strange = set()
	strange.add("\x81")
	strange.add("“")
	strange.add("”")
	strange.add("‘")
	strange.add("..")
	strange.add("🤣")
	strange.add("..")
	strange.add("...")
	strange.add("°")
	strange.add("…")
	strange.add("\x89")
	strange.add("ĚĄ")
	strange.add("ă")
	strange.add("\x9d")
	strange.add("âÂĺ")
	strange.add("Ě")
	strange.add("˘")
	strange.add("Â")
	strange.add("âÂ")
	strange.add("Ň")
	strange.add("000")
	strange.add("Ň")
	strange.add("Ã")
	strange.add("Ã¹")
	strange.add("Ã")
	strange.add("¹")
	strange.add("â€™")
	strange.add("™")
	strange.add("Ã²")
	strange.add("²")
	strange.add("Ã")
	strange.add("Ã")
	strange.add("Ã¬")
	strange.add("¬")
	strange.add("\";(\"")
	strange.add("’")
	strange.add("€")
	strange.add("❌")
	strange.add("☆")
	strange.add("★")
	strange.add("🖤")
	strange.add("💲")
	strange.add("💣")
	strange.add("ľ")
	strange.add("😂")
	strange.add("👍")
	strange.add("🏻")
	strange.add("💯")
	strange.add("👌")
	strange.add("‼")
	strange.add("️")
	strange.add(":d")
	strange.add("😱")
	strange.add("😉")
	strange.add("😍")
	strange.add("💪")
	strange.add("🔝")
	strange.add("〽")
	strange.add("💙")
	strange.add("⚪")
	strange.add("⚫")
	strange.add("👎")
	strange.add("\";)\"")
	strange.add("\";-)\"")
	strange.add("😊")
	strange.add("❤")
	strange.add("😭")
	strange.add("📦")
	strange.add("✅")
	strange.add("🚧")
	strange.add("・")
	strange.add("😐")
	strange.add("->")
	strange.add("▶")
	strange.add("▸")
	strange.add("📝")
	strange.add("🏁")
	strange.add("✔")
	strange.add("❗")
	strange.add("👏")
	strange.add("👋")
	strange.add("🙋")
	strange.add(" ")
	strange.add("♀")
	strange.add("️grazie")
	strange.add("💩")
	strange.add("🙋")
	strange.add("⚽")
	strange.add("\";]\"")
	strange.add("😘")
	strange.add("●")
	strange.add("○")
	strange.add("😩")
	strange.add("⭐")
	strange.add("●")
	return list(strange)

def main():

	# data.dtypes

	# Columns access
	# data["_id"] 
	# data._id

	# Rows access
	# data.iloc[0]
	# data[20:30]
	# data.iloc[20:30]
	# data.loc[data._id == "B07PYG9BB4"]

	# data.head()

	with open('products.json', 'r', encoding='utf8') as f:
		data = pd.DataFrame(json.loads(line) for line in f)
	
	# List of split words and symbols
	#print(word_tokenize(data.iloc[0].description))

	# List of split words and symbols with regex
	#print(preprocess(data.iloc[0].description))

	'''
	tokening = TweetTokenizer()
	s0 = "@remy This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
	print("Twitter tokenizer -> ",tokening.tokenize(s0))
	print("Preprocess function -> ",preprocess(s0))
	'''

	tokening = TweetTokenizer(strip_handles=True, reduce_len=True)
	# strip_handles = True
	# Remove the people tag it's useful in sentiment analysis
	# reduce_len = True
	# If person written more (or equal)	than 3 same letter than we'll see only 3 letters
	#s1 = '@remy: This is waaaaayyyy too much for youuuuuu!!!!!!'
	#print("Twitter tokenizer -> ",tokening.tokenize(s1))

	# Apply in multiprocessing function preprocess to all rows
	#data.description.apply(preprocess)
	data_tokenized = data["description"].apply(tokening.tokenize)

	# get_counter Without removing Stopword and Punctuation
	#c = get_counter(data_tokenized)
	#print(c.most_common(10))
	
	# Warning, these words could be useful when you do sentiment analysis, so be attention
	# to remove Stopword and Punctuation. Obv, first remove Stopword and then Punctuation.
	stop = addStopwords()
	punctuation = string.punctuation
	data_tokenized_stop = data_tokenized.apply(lambda x: [item for item in x if item not in stop])
	data_tokenized_stop_punct = data_tokenized_stop.apply(lambda x: [item for item in x if item not in punctuation])
	
	c = get_counter(data_tokenized_stop_punct)
	print(c.most_common(10))

	'''
	# IS THIS REALLY HELPFUL?
	# Stemming is the process of reducing inflected (or sometimes derived) words to their word stem,
	# base or root form—generally a written word form.

	from nltk.stem.lancaster import LancasterStemmer
	lancaster_stemmer = LancasterStemmer()

	data_tokenized_new_stem = data_tokenized_stop_punct.apply(lambda x: [lancaster_stemmer.stem(item) for item in x])
	'''
	
	# Visualize words
	visualizeWords(data_tokenized_stop_punct)

	'''
	# ONLY WORK WITH ENGLISH WORDS!!!
	# Sentiment Analysis - Lexicon-based approach
	# The AFINN lexicon is a list of English terms manually rated for 
	# valence with an integer between -5 (negative) and +5 (positive) 
	# by Finn Årup Nielsen between 2009 and 2011.
	afinn = Afinn()
	sentences = ["There is a terrible mistake in this work", "This is wonderful!"]
	s2 = [afinn.score(s) for s in sentences]
	print(s2)
	'''

# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()

# Helper function
def print_topics(model, count_vectorizer, n_top_words):
	words = count_vectorizer.get_feature_names()
	for topic_idx, topic in enumerate(model.components_):
		print("\nTopic #%d:" % topic_idx)
		print(" ".join([words[i]
						for i in topic.argsort()[:-n_top_words - 1:-1]]))

def project():
	# CERCA SU GOOGLE: python library for Aspet Sentiment

	with open('products.json', 'r', encoding='utf8') as f:
		data = pd.DataFrame(json.loads(line) for line in f)

	tokening = TweetTokenizer(strip_handles=True, reduce_len=True)
	data_tokenized = data[0:100]["title"].apply(tokening.tokenize)

	stop = addStopwords()
	punctuation = string.punctuation
	data_tokenized_stop = data_tokenized.apply(lambda x: [item for item in x if item not in stop])
	data_tokenized_stop_punct = data_tokenized_stop.apply(lambda x: [item for item in x if item not in punctuation])
	
	# Visualize words
	visualizeWords(data_tokenized_stop_punct)

	# LDA
	# https://github.com/kapadias/mediumposts/blob/master/nlp/published_notebooks/Introduction%20to%20Topic%20Modeling.ipynb
	# Initialise the count vectorizer with the English stop words
	count_vectorizer = CountVectorizer(stop_words=None, lowercase=True, max_features=5000)

	# Fit and transform the processed titles
	count_data = count_vectorizer.fit_transform(data_tokenized_stop_punct.apply(lambda x: " ".join(x)))

	# Visualise the 10 most common words
	plot_10_most_common_words(count_data, count_vectorizer)

	# Tweak the two parameters below
	number_topics = 5
	number_words = 10
	# Create and fit the LDA model
	lda = LDA(n_components=number_topics, n_jobs=-1)
	lda.fit(count_data)
	# Print the topics found by the LDA model
	print("Topics found via LDA:")
	print_topics(lda, count_vectorizer, number_words)

	# Analyzing LDA model results
	# Visualize the topics
	LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
	pyLDAvis.save_html(LDAvis_prepared, './ldavis_prepared_'+ str(number_topics) +'.html')
	pyLDAvis.show(LDAvis_prepared)

emoticons_str = r"""
(?:
	[:=;] # Eyes
	[oO\-]? # Nose (optional)
	[D\)\]\(\]/\\OpP] # Mouth
)"""

regex_str = [
	emoticons_str,
	r'<[^>]+>', # HTML tags
	r'(?:@[\w_]+)', # @-mentions
	r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
	r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

	r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
	r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
	r'(?:[\w_]+)', # other words
	r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
	return tokens_re.findall(s)

def preprocess(s, lowercase=False):
	tokens = tokenize(s)
	if lowercase:
		tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
	return tokens

def regexCheck(item):
	a = re.search("[0-9]", item)
	b = re.search("'", item)
	c = re.search("[_]", item)
	d = re.search("[.]", item)
	return a or c or d

def reviews():
	with open('reviews.json', 'r', encoding='utf8') as f:
		data_REVIEWS = pd.DataFrame(json.loads(line) for line in f)
	data_REVIEWS.set_index('product', inplace=True)

	with open('products.json', 'r', encoding='utf8') as f:
		data_PRODUCTS = pd.DataFrame(json.loads(line) for line in f)
	data_PRODUCTS.set_index('_id', inplace=True)

	data = data_REVIEWS.join(data_PRODUCTS, how='left', lsuffix='_left', rsuffix='_right')

	data = data.loc[(data.verified==True) & (data.category=="videogames")]

	tokening = TweetTokenizer(strip_handles=True, reduce_len=True)
	#document = data[0:100]["body"].apply(nltk.sent_tokenize)

	print("Now cut all sentences")
	document = data[0:10000]["body"].apply(nltk.sent_tokenize)
	print("All sentences are cut", "\n")

	stop = addStopwords()
	strange = addStranges()
	punctuation = string.punctuation

	#print("Number of empty sentences: " + str(len([1 for review in document for sentence in review if len(sentence)==0])), "\n")

	bagOfSentences = []
	wordsList = []
	count = 0
	sizeDocument = len(document)
	for review in document:
		bagOfSentences.append(len(review))
		indexNumOfSentences = len(bagOfSentences)-1
		for sentence in review:
			tmpC = []
			words = tokening.tokenize(sentence)
			words = [item.lower() for item in words if item not in punctuation and item not in strange and item.lower() not in stop and not regexCheck(item)] # for cleaning
			if len(words)>300 or len(words)<5:
				bagOfSentences[indexNumOfSentences] = bagOfSentences[indexNumOfSentences]-1
				continue
			# create wordsList
			for word in words:
				wordsList.append(word)
				tmpC.append(len(wordsList)-1)
				for index, item in enumerate(wordsList[:-1]):
					if word == item:
						del tmpC[-1]
						tmpC.append(index)
						del wordsList[-1]
						break
			bagOfSentences.append(' '.join(str(e) for e in tmpC))
		print(sizeDocument - count)
		count = count + 1

	#print("Number of empty sentences after processing: " + str(len([1 for review in document for sentence in review if len(sentence)==0])), "\n")

	wordsList = pd.DataFrame(wordsList)
	bagOfSentences = pd.DataFrame(bagOfSentences)
	wordsList.to_csv('WordList.txt', header=None, index=None, sep=";")
	bagOfSentences.to_csv('BagOfSentences.txt', header=None, index=None, quoting=csv.QUOTE_NONE, escapechar = " ")

if __name__ == "__main__":
	#project()
	reviews()

