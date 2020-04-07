import argparse
import os
import re
import nltk
import pickle
from nltk.stem import PorterStemmer 

import numpy as np
from numpy import dot
from numpy.linalg import norm

def cosine(vector1, vector2):
    return float(dot(vector1, vector2) / (norm(vector1) * norm(vector2)))

def euclidean(vector1, vector2):
    return norm(np.array(vector1) - np.array(vector2))

def remove_stop_words(token_list):
    # Stopword Dictionary
    stopwords_dict = open('./english.stop', 'r').read().split()   
    # Stemmer
    ps = PorterStemmer() 
    return [ps.stem(word) for word in token_list if word not in stopwords_dict and word != ""]

def clean(sentence):
    sentence = sentence.replace(".", "")
    sentence = sentence.replace("\s+", " ")
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = sentence.lower()
    return sentence

def search(query_vector, collection, way, freq):
    rating_dict = {}
    for product_id, value in collection.items():
        if freq == "TF": vector = value[1]
        elif freq == "TF-IDF": vector = value[2]
        if way == "cos": simlarity = cosine(query_vector, vector)
        elif way == "geo": simlarity = euclidean(query_vector, vector)
        rating_dict[product_id] = simlarity
        
    if way == "cos":
        top_5 = sorted(rating_dict.items(), key=lambda kv: kv[1], reverse=True)[:5]
    elif way == "geo":
        top_5 = sorted(rating_dict.items(), key=lambda kv: kv[1], reverse=False)[:5]
        
    for items in top_5:
        print(items[0], round(items[1], 5))
    print("============================")
    
    return top_5

if __name__ == '__main__':
    
    ### Parser Argument ###
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required = True, nargs='*')
    input_query = parser.parse_args().query
    ##########################
    
    ### Read Stored Files ###
    # Tf-Idf Vector Dictionary
    with open('./tfidf_vector.pickle', 'rb') as handle:
        tf_idf_collection = pickle.load(handle)
    # Word Dictionary
    with open('./word_index.pickle', 'rb') as handle:
        word_index = pickle.load(handle)
    # Idf Vector Dictionary
    with open('./idf_vector.pickle', 'rb') as handle:
        idf_collection = pickle.load(handle)
    ##########################
    
    ### Process Query ###
    query = " ".join(input_query)
    clean_query = clean(query)
    token_query = clean_query.split(" ")
    filtered_query = remove_stop_words(token_query)
    ##########################
    
    ### Query to Vector ###
    freq_dict = nltk.FreqDist(filtered_query)
    query_tf_vector = [0] * len(word_index)
    query_tf_idf_vector = [0] * len(word_index)

    for word, freq in freq_dict.items():
        query_tf_vector[word_index[word]] = freq
        query_tf_idf_vector[word_index[word]] = freq * idf_collection[word]   
    ##########################
    
    ### Search top-5 (Q1) ###
    print("TF Weighting + Cosine Similarity:")
    print()
    Q1_top_5 = search(query_tf_vector, tf_idf_collection, "cos", "TF")
    ##########################
    
    ### Search top-5 (Q2) ###
    print("TF Weighting + Euclidean Distance:")
    print()
    Q2_top_5 = search(query_tf_vector, tf_idf_collection, "geo", "TF")
    ##########################
    
    ### Search top-5 (Q3) ###
    print("TF-IDF Weighting + Cosine Similarity:")
    print()
    Q3_top_5 = search(query_tf_idf_vector, tf_idf_collection, "cos", "TF-IDF")
    ##########################
    
    ### Search top-5 (Q4) ###
    print("TF-IDF Weighting + Euclidean Distance:")
    print()
    Q4_top_5 = search(query_tf_idf_vector, tf_idf_collection, "geo", "TF-IDF")
    ##########################
    
    ### Search top-5 (Q5) ###
    print("Feedback Queries + TF-IDF Weighting + Cosine Similarity:")
    print()
    confirmed_tag = ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    
    # Keep Noun and Verb
    top_1_number = Q3_top_5[0][0]
    top_1_token = []
    for tag in nltk.pos_tag(tf_idf_collection[top_1_number][0]):
        if tag[1] in confirmed_tag:
            top_1_token.append(tag[0])
            
    # Redo the TF-IDF vector
    top_1_freq_dict = nltk.FreqDist(top_1_token)
    top_1_tf_idf_vector = [0] * len(word_index)
    for word, freq in top_1_freq_dict.items():
        top_1_tf_idf_vector[word_index[word]] = freq * idf_collection[word]
    
    # New Query Tf-IDF vector (1 * original query + 0.5 * feedback query)
    feedback_tf_idf_vector = np.array(query_tf_idf_vector) + 0.5 * np.array(top_1_tf_idf_vector)
    Q5_top_5 = search(feedback_tf_idf_vector, tf_idf_collection, "cos", "TF-IDF")
    ##########################
    