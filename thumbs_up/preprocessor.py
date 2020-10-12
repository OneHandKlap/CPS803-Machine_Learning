import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string

class Preprocessor(object):

    def __init__(self,dataframe):
        self.data=dataframe

    #specify column title to be tokenized with column
    def tokenize(self,column):
        tokenizer = RegexpTokenizer('\w+')
        self.data[column]=(self.data[column].apply(tokenizer.tokenize))
    
    def lemmatize(self,column):
        stop_words=stopwords.words('english')
        stop_words.append('br')
        def lemmatize_tokens(tokens):
            lemmatizer=WordNetLemmatizer()
            output=[]

            for word, tag in tokens:
                word = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                            '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', word)
                word = re.sub("(@[A-Za-z0-9_]+)","", word)
                word = re.sub("[0-9]+","",word)
                if tag.startswith('NN'):
                    pos='n'
                elif tag.startswith('VB'):
                    pos='v'
                else:
                    pos='a'
                if word.lower() not in stop_words and word not in string.punctuation:
                    output.append(lemmatizer.lemmatize(word.lower(),pos))
            return output
        self.data[column]=self.data[column].apply(lemmatize_tokens)
        print(self.data)

def main(dataframe):
    dataframe=pd.read_csv(dataframe, names=['review','sentiment'])
    preprocessor=Preprocessor(dataframe)
    
    preprocessor.tokenize('review')

    preprocessor.lemmatize('review')


if __name__=='__main__':
    main('small1.csv')