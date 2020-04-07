import os
import re
import nltk
import pickle
import numpy as np
from nltk.stem import PorterStemmer 

def clean(doc):
    doc = doc.replace(".", "")
    doc = doc.replace("\s+", " ")
    doc = re.sub('[^A-Za-z]+', ' ', doc)
    doc = doc.lower()
    return doc

def remove_stop_words(token_list):
    # Stopword Dictionary
    stopwords_dict = open('./english.stop', 'r').read().split()   
    # Stemmer
    ps = PorterStemmer() 
    return [ps.stem(word) for word in token_list if word not in stopwords_dict and word != ""]

if __name__ == '__main__':
    
    ### Reading All Doc ###
    text_collection = []
    doc_id_list = []
    for index, filename in enumerate(os.listdir("./documents/")):
        doc_id_list.append(filename.replace(".product", ""))
        with open(os.path.join("./documents/", filename), 'r') as f:
            text = f.read()
            text_collection.append(text)
    ##########################
    
    ### Index All Words ###
    word_index = {}
    all_raw_doc = " ".join(text_collection)
    # Clean Word
    clean_all_doc = clean(all_raw_doc)
    # Tokenize Word
    token_all_doc = clean_all_doc.split(" ")
    # Remove Stopwords and Stemming
    filtered_all_doc = remove_stop_words(token_all_doc)
    # Find the Word Set
    unique_token_all_doc = list(set((item for item in filtered_all_doc)))
    # Create Word Index
    for index, unique_token in enumerate(unique_token_all_doc):
        word_index[unique_token] = index
    ##########################
    
    ### All Sentence in Doc ###
    clean_token_collection = []
    for raw_doc in text_collection:
        # Clean Word
        clean_doc = clean(raw_doc)
        # Tokenize Word
        token_doc = clean_doc.split(" ")
        # Remove Stopwords and Stemming
        filtered_doc = remove_stop_words(token_doc)
        clean_token_collection.append(filtered_doc)
    ##########################
    
    ### IDF Calculation ###
    idf_collection = {}
    # Initialize IDF Dictionary
    for word in word_index.keys():
        idf_collection[word] = 0
        
    # IDF: log ( N / df_i )
    for doc_token in clean_token_collection:
        for word in word_index.keys():
            if word in doc_token:
                idf_collection[word] += 1
    for word, value in idf_collection.items():
        idf_collection[word] = np.log(len(clean_token_collection) / value)
    ##########################
    
    ### TF & TF-IDF Calculation ###
    # Save Vector into a Dictionary
    id_vector_dict = {}
    for doc_id, doc_token in zip(doc_id_list, clean_token_collection):
        id_vector_dict[doc_id] = [doc_token, [], []]
    
    for n, doc_token in enumerate(clean_token_collection):
        prodict_number = str(doc_id_list[n])
        freq_dict = nltk.FreqDist(doc_token)
        tf_vector = [0] * len(word_index)
        tf_idf_vector = [0] * len(word_index)
        for word, freq in freq_dict.items():
            tf_vector[word_index[word]] = freq
            tf_idf_vector[word_index[word]] = freq * idf_collection[word]
        id_vector_dict[prodict_number][1] = tf_vector
        id_vector_dict[prodict_number][2] = tf_idf_vector
    ##########################
    
    ### Save what I need in Query ###
    with open('./tfidf_vector.pickle', 'wb') as handle:
        pickle.dump(id_vector_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./word_index.pickle', 'wb') as handle:
        pickle.dump(word_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./idf_vector.pickle', 'wb') as handle:
        pickle.dump(idf_collection, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ##########################