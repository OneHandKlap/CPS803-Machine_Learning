import pandas as pd
import numpy as np
from bayes_model import BayesModel
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk import FreqDist
import string
from sklearn.metrics import confusion_matrix
import model_analyzer


#purpose of this module is to make it straightforward and simple to conduct preprocessing for NLP projects
class Preprocessor(object):

    #initialize preprocessor object by passing in a Pandas dataframe
    def __init__(self,dataframe,vocabulary=None):
        self.data=dataframe
        self.vocabulary=vocabulary

    #specify column title to be tokenized
    #tokenization refers to the 'splitting up' of individual words within a block of text
    def tokenize(self,column):
        tokenizer = RegexpTokenizer('\w+')
        self.data[column]=(self.data[column].apply(tokenizer.tokenize))
    
    #lemmatization refers to the reformatting of tokenized words
    #ie. removing 'stop words', punctuation, numbers etc.
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

    #this function adds identifiers to individual words which indicates their 'part of speech'
    def add_tags(self,column):
        self.data[column]=self.data[column].apply(pos_tag)

    #takes an integer to determine the number of words we want to use per text in order to contribute to our overall vocabulary
    def create_vocabulary(self, column, num_words_per_text):
        vocabulary=[]
        for lemmatized_text in self.data[column]:
            freq_dist=FreqDist(lemmatized_text)
            count=0
            for word,times in freq_dist.most_common(50):
                if count==10:
                    break
                elif word not in vocabulary and times>1:
                    vocabulary.append(word)
                    count+=1
        self.vocabulary= vocabulary

    #once a vocabulary has been established, this function recreates the dataframe
    #it creates a specific column for each word in the vocabulary and includes the number of times that word
    #is found in each text
    def update_dataframe(self,text_column,y_column):
        new_df=pd.DataFrame()
        new_df['x']=self.data[text_column]
        
        for j in range(len(self.vocabulary)):
            new_df['x'+str(j+1)]=[None]*len(new_df)
            for i in range(len(new_df['x'])):
                
                count=0
                for word in new_df['x'][i]:

                    if word==self.vocabulary[j]:
                        count+=1

                new_df['x'+str(j+1)][i]=count
        def make_binary(entry):
            if entry=='positive':
                return 1
            else:
                return 0
        new_df['y']=self.data[y_column].apply(make_binary)
        self.data=new_df
def make_binary(entry):
    if entry=='positive':
        return 1
    else:
        return 0

def main(train_path,test_path):
    
    train_df=pd.read_csv(train_path, names=['x','y'])
    preprocessor=Preprocessor(train_df)
    
    print("PROCESSING TRAINING DATA")
    preprocessor.tokenize('x')
    preprocessor.add_tags('x')
    preprocessor.lemmatize('x')
    print("CREATING MODEL VOCABULARY")
    preprocessor.create_vocabulary('x',10)
    print("UPDATING TRAINING DATAFRAME")
    preprocessor.update_dataframe('x','y')
    
    model=BayesModel(preprocessor.vocabulary)
    print("FITTING MODEL")
    model.fit_laplace(preprocessor.data)
    analyzer=model_analyzer.Analyzer(model,test_path)
    thresholds=[round((0.05*x),2) for x in range(20,-1,-1)]
    
    analyzer.threshold_scan(thresholds,'threshold.csv')


    # print("USING MODEL TO PREDICT SENTIMENT")
    # preprocessor.data['pos_score'],preprocessor.data['neg_score']=(model.predict(preprocessor.data))
    # preprocessor.data.to_csv('result.csv')

    # print("SCORE: "+str(len(preprocessor.data['results'].loc[preprocessor.data['results']==preprocessor.data['y']])/len(preprocessor.data)))
    




if __name__=='__main__':
    main('small1.csv','test.csv')