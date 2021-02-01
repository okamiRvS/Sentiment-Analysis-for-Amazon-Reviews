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
from nltk.stem import SnowballStemmer

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

import treetaggerwrapper 
from pprint import pprint

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
	stop.add("essere")
	stop.add("avere")
	stop.add("not_essere")
	stop.add("not_avere")
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
	#b = re.search("'", item)
	#c = re.search("[_]", item)
	d = re.search("[.]", item)
	return a or d

def reviews():
	# List Unique Values In A pandas Column
	# data.category.unique()

	# How to select rows from a DataFrame based on column values
	# data.loc[data["category"] == "pc"]


	with open('products.json', 'r', encoding='utf8') as f:
		data_PRODUCTS = pd.DataFrame(json.loads(line) for line in f)
	data_PRODUCTS.set_index('_id', inplace=True)

	data_PRODUCTS.to_pickle('cached_data_PRODUCTS.pkl')

	with open('reviews.json', 'r', encoding='utf8') as f:
		data_REVIEWS = pd.DataFrame(json.loads(line) for line in f)
	data_REVIEWS.set_index('product', inplace=True)

	data_BOOKS = data_PRODUCTS.loc[data_PRODUCTS["category"] == "books" ]
	data_join_BOOKS = data_REVIEWS.join(data_BOOKS,  how='inner', lsuffix='_left', rsuffix='_right')
	data_join_BOOKS["index1"] = data_join_BOOKS.index
	data = data_join_BOOKS
	
	# Store your DataFrame
	data.to_pickle('cached_dataframe.pkl') # will be stored in current directory


	# Tree-tagger installation. When you're installing tree-tagger, you have 3 big steps :
	# 	1 - Download tree-tagger
	# 	2 - Download the scripts into your tree-tagger directory
	# 	3 - Then use the install-tagger.sh
	# 	give a TAGDIR named argument when building a TreeTagger object to provide this information: tagdir = ….
	tagger = treetaggerwrapper.TreeTagger(TAGLANG="it", TAGDIR="C:/TreeTagger/")
	'''
	it_string = "Ieri sono andato in due supermercati. Oggi volevo andare all'ippodromo. Stasera mangio la pizza con le verdure."
	tags = tagger.tag_text(it_string)
	pprint(treetaggerwrapper.make_tags(tags))
	'''

	# stemming for Italian language
	stemmer_snowball = SnowballStemmer('italian')
	'''
	# create two example word-lists
	eg1 = ['correre', 'corro', 'corriamo', 'correremo']
	eg2 = ['bambino', 'bambini', 'bambina', 'bambine']
	print(stemmer_snowball.stem(eg1[0]))
	print(stemmer_snowball.stem(eg2[0]))
	'''

	# Read your DataFrame
	data = pd.read_pickle('cached_dataframe.pkl') # read from current directory
	data_PRODUCTS = pd.read_pickle('cached_data_PRODUCTS.pkl')

	# dataframe to graph

	import igraph as ig

	qwerty = data.drop_duplicates("index1") # 906 unique books
	
	# G.vs[0]["name"]
	# G.neighbors(0)

	G = ig.Graph(directed=True)
	for node in data_PRODUCTS.index:
		G.add_vertices([node])

	labels = []
	weight = []
	for node in data_PRODUCTS.index:
		bought_together_nodes = data_PRODUCTS.loc[node]["bought_together"]
		also_bought_nodes = data_PRODUCTS.loc[node]["also_bought"]
		also_viewed_nodes = data_PRODUCTS.loc[node]["also_viewed"]
		for bought_node in bought_together_nodes:
			w = 0.5
			flag_A = False
			flag_B = False
			for also_bought in also_bought_nodes:
				if bought_node == also_bought :
					w = w + 0.3
					flag_A = True
			for viewed_nodes in also_viewed_nodes:
				if bought_node == viewed_nodes :
					w = w + 0.2
					flag_B = True
			try:
				G.add_edges([(node, bought_node)])
				weight.append(w)
				labels.append(str(w))
				if flag_A : 
					also_bought_nodes.remove(bought_node)
				if flag_B :
					also_viewed_nodes.remove(bought_node)
			except Exception as e:
				print("bought_node: " + str(e))

		for also_bought in also_bought_nodes:
			w = 0.3
			flag_A = False
			for viewed_nodes in also_viewed_nodes:
				if also_bought == viewed_nodes :
					w = w + 0.2
					flag_A = True
			try:
				G.add_edges([(node, also_bought)])
				weight.append(w)
				labels.append(str(w))
				if flag_A :
					also_viewed_nodes.remove(also_bought)
			except Exception as e:
				print("also_bought: " + str(e))

		for viewed_nodes in also_viewed_nodes:
			try:
				G.add_edges([(node, viewed_nodes)])
				weight.append(0.2)
				labels.append("0.2")
			except Exception as e:
				print("viewed_nodes: " + str(e))

	G.es["weight"] = labels
	G.es["label"] = labels

	# INDEGREE
	# this is a way to quantify the importance, but
	# with indegree measure we don't take into account the quality of each of these links,
	# a link coming form W might not mean as much as a link coming from Z, because
	# Z is more popular than W
	# https://www.coursera.org/lecture/networks-illustrated/in-degree-IlnXY
	indegree_list = G.indegree()
	n_argmax = np.argsort(indegree_list)
	n_argmax_reverse = n_argmax[::-1]
	max_indegree = indegree_list[n_argmax_reverse[0]]

	print("\n \n INDEGREEEEEEEEEE")
	indegree_books_values = []
	count = 0
	for value in n_argmax_reverse :
		id_product = G.vs[value]["name"]
		for book_id in qwerty.index :
			if id_product == book_id and count < 30:
				indegree_books_values.append(indegree_list[value])
				print(data_PRODUCTS.loc[id_product].title + " " + id_product)
				count += 1
				continue
	
	# PAGERANK
	print("\n \n PAGERANKKK")
	pagerank_list = G.pagerank()
	n_argmax = np.argsort(pagerank_list)
	n_argmax_reverse = n_argmax[::-1]

	pagerank_list_books_values = []
	count = 0
	for value in n_argmax_reverse :
		id_product = G.vs[value]["name"]
		for book_id in qwerty.index :
			if id_product == book_id and count < 30:
				pagerank_list_books_values.append(pagerank_list[value])
				print(data_PRODUCTS.loc[id_product].title + " " + id_product)
				count += 1
				continue

	# statistics of the graph

	print("Nodes: " + str(len(G.vs)))
	print("Edges: " + str(len(G.es)))
	print("Directed: " + "True")
	print("Density: " + str(G.density(loops=False)))
	print("Reciprocity: " + str(G.reciprocity(ignore_loops=True, mode="default"))) # Reciprocity defines the proportion of mutual connections in a directed graph.
	print("Assortativity: " + str(G.assortativity_degree(directed=True)))
	print("Average degree: " + str(G.average_path_length(directed=True, unconn=True)))
	print("Max Indegree: " + str(max_indegree))

	#print("Max Outdegree : " + G.diameter(directed=True, unconn=True))
	#print("Diameter: " + G.density(loops=False))
	#print("Connected components: " + G.density(loops=False))
	#print("Giant component dimension: " + len(G.clusters().giant().vs))


	# G.neighbors(2)
	# G.es["weight"][0]
	# ig.plot(G)

	# get number of recension of each "il piccolo principe" book
	#data.loc[data["title_right"] == "Il Piccolo Principe"].groupby(["index1"])
	qwerty = data.groupby(["index1", "title_right"])["_id"].nunique().sort_values(ascending=False)
	qwerty = qwerty.reset_index() # for convert MultiIndex to columns
	qwerty = qwerty.rename({"_id" : "count"}, axis="columns")
	qwerty = qwerty.loc[qwerty["title_right"] == "Il Piccolo Principe"]
	qwerty = qwerty.join(data,  how='left', on="index1" , lsuffix='_left', rsuffix='_right')
	qwerty = qwerty.drop_duplicates("index1")
	#qwerty.iloc[0].date
	qwerty["x_labels"] = qwerty["title_right_right"] + "\n" + qwerty["date"].astype(str)
	# creating the bar plot 
	ax = qwerty.plot.bar(x="x_labels", y="count", color=(.4, .5, .6), rot=0, zorder=3)
	ax.grid(zorder=0)
	ax.set_xlabel("")
	ax.set_ylabel("n° of reviews")
	plt.show()

	# Choosing the most interesting book
	data.groupby("index1")["_id"].nunique().sort_values(ascending=False)

	# All review of "il piccolo principe" of this product 8854172383
	data = data.loc[(data.index1 == "8854172383") & (data.verified == True)]
	data["body"] =  data["title_left"] + ". " + data["body"] # join title review with body review
	bars = ["bad \n rating <= 3", "good \n rating > 3"]
	pos_rating_review = len([d.get("$numberInt") for d in data.rating if int(d.get("$numberInt")) > 3])
	neg_rating_review = len([d.get("$numberInt") for d in data.rating if int(d.get("$numberInt")) <= 3])
	value_bars = [neg_rating_review, pos_rating_review]
	plt.bar(x=bars, height=value_bars, color=(.4, .5, .6), zorder=3)
	plt.grid(zorder=0)
	plt.xlabel("")
	plt.ylabel("n° of stars")
	#plt.show()

	# Tokenization - Now we cut all sentences
	tokening = TweetTokenizer(strip_handles=True, reduce_len=True)
	#document = data[0:100]["body"].apply(nltk.sent_tokenize)
	document = data["body"].apply(nltk.sent_tokenize)

	# Negation handling
	# not_list = ["non", "ma", "però", "invece", "anzi", "bensì", "tuttavia", "nondimeno", "pure", "eppure"]
	not_list = ["non"]
	# for review in document 
	for z in range(len(document)) :
		# for sentence in review
		for j in range(len(document[z])) :
			x = document[z][j].split(" ")
			not_activate = False
			# for word in sentence
			for i in range(len(x)) :
				adj = False
				# POS tagging, so "andato" -> "andare"
				try:
					# only if aggettivo
					tags = tagger.tag_text(x[i])
					if "adj" in tags[0].lower() :
						adj = True
					x[i] = tags[0].split("\t")[2] #  lemmatizing
				except Exception as e:
					print(str(e))
				'''
				# Stemming, so "correre" -> "corr"
				try:
					x[i] = stemmer_snowball.stem(x[i])
				except Exception as e:
					print(str(e))	
				'''
				if adj and not_activate and len(x[i]) > 3:
					x[i] = "not_{}".format(x[i])
				else :
					for p in range(len(not_list)) :
						if not_list[p] == x[i] :
							not_activate = True
							break
			document[z][j] = ' '.join(e for e in x if len(e)>3)

	stop = addStopwords()
	strange = addStranges()
	punctuation = string.punctuation
	# print("Number of empty sentences: " + str(len([1 for review in document for sentence in review if len(sentence)==0])), "\n")

	#data = data.rename({"body" : "body_review"}, axis="columns")
	#document = pd.concat([data, document], axis=1)

	good_review = []
	bad_review = []
	for index, review in data.iterrows() : 
		if int(review.rating.get("$numberInt")) > 3 :
			good_review.append(review["body"])
		else :
			bad_review.append(review["body"])
	
	print("Number of bad_review is: {}".format(len(bad_review)))
	print("Number of good review is: {}".format(len(good_review)))

	
	# Visualize bad words
	df = pd.DataFrame(bad_review)

	data_tokenized = df[0].apply(tokening.tokenize)
	data_tokenized_stop = data_tokenized.apply(lambda x: [item for item in x if item.lower() not in stop])
	data_tokenized_stop_punct = data_tokenized_stop.apply(lambda x: [item for item in x if item.lower() not in punctuation])
	c = get_counter(data_tokenized_stop_punct)
	print(c.most_common(10))
	visualizeWords(data_tokenized_stop_punct)


	# Visualize good words
	df = pd.DataFrame(good_review)

	data_tokenized = df[0].apply(tokening.tokenize)
	data_tokenized_stop = data_tokenized.apply(lambda x: [item for item in x if item.lower() not in stop])
	data_tokenized_stop_punct = data_tokenized_stop.apply(lambda x: [item for item in x if item.lower() not in punctuation])
	c = get_counter(data_tokenized_stop_punct)
	print(c.most_common(10))
	visualizeWords(data_tokenized_stop_punct)


	# words for sentiment analysis	
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

	# transform good words in bad words adding not_ prefix
	# https://github.com/gragusa/sentiment-lang-italian
	f = open("SentiWords-0.txt", "r")
	a = open("SentiWords-0_not.txt", "w")
	for x in f:
		a.write("not_" + x)
	f.close()
	a.close()

	f = open("SentiWords-1.txt", "r")
	a = open("SentiWords-1_not.txt", "w")
	for x in f:
		a.write("not_" + x)
	f.close()
	a.close()


if __name__ == "__main__":
	reviews()
