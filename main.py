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
	reviews()
