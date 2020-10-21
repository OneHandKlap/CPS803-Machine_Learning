import pandas as pd
import numpy as np


class BayesModel(object):

    def __init__(self,vocabulary,thetas=None):
        self.vocab=vocabulary
        if thetas==None:
            self.theta=[]
        else:
            self.theta=thetas
            
    def fit_laplace(self,train_df):
        total_positives=sum(train_df['y']==1)+2
        for col in (train_df.columns):
            if col not in ['x','y']:
                count_word=sum(train_df[col].loc[train_df['y']==1])+1
                probability_word=count_word/total_positives
                self.theta.append(probability_word)


    def fit(self,train_df):

        total_positives=sum(train_df['y']==1)
        for col in (train_df.columns):
            if col not in ['x','y']:
                count_word=sum(train_df[col].loc[train_df['y']==1])
                probability_word=count_word/total_positives
                self.theta.append(probability_word)
        

    def predict(self,test_df):

        count=1
        for col in (test_df.columns):
            if col not in ['x','y']:
                test_df[col]=test_df[col]*self.theta[count]
                count+=1
        
        def total_probability(row):
            total=1
            for i in range(len(self.thetas)):
                total=total*row['x'+str(i+1)]
            return total

        return test_df.apply(total_probability,axis=1)

def main(dataframe_path,train_path):

    model=BayesModel()
    model.fit(dataframe_path)


if __name__=='__main__':
    main('converted_small.csv')