import pandas as pd
import numpy as np


class BayesModel(object):

    def __init__(self,vocabulary,thetas=None):
        self.vocab=vocabulary
        if thetas==None:
            self.theta=[[],[]]
        else:
            self.theta=thetas


    def fit_laplace(self,train_df):

        for col in (train_df.columns):
            if col not in ['x','y']:
                count_word_pos=sum(train_df[col].loc[train_df['y']==1])
                count_word_neg=sum(train_df[col].loc[train_df['y']==0])
                probability_word_pos=count_word_pos+1/(sum(train_df['y']==1)+2)
                probability_word_neg=count_word_neg+1/(sum(train_df['y']==0)+2)
                self.theta[0].append(probability_word_pos)
                self.theta[1].append(probability_word_neg)

        print(self.theta)

                

    def fit(self,train_df):

        #total_positives=sum(train_df['y']==1)
        for col in (train_df.columns):
            if col not in ['x','y']:
                count_word_pos=sum(train_df[col].loc[train_df['y']==1])
                count_word_neg=sum(train_df[col].loc[train_df['y']==0])
                probability_word=count_word_pos/(count_word_neg+count_word_pos)
                self.theta.append(probability_word)

    def get_top_five(self):
        
        
        def GetKey(val,dicti):
            for key, value in dicti.items():
                if val == value:
                    return key
        
        
        top_five={}
        bot_five={}
        for i in range(len(self.theta[0])):

            prob_pos=self.theta[0][i]/(self.theta[0][i]+self.theta[1][i])
            prob_neg=1-prob_pos
            if len(top_five)<5:
                top_five[self.vocab[i]]=prob_pos
            if len(bot_five)<5:
                bot_five[self.vocab[i]]=prob_neg
            if prob_neg>min(bot_five.values()):
                key = GetKey(min(bot_five.values()),bot_five)
                bot_five.pop(key)
                
                bot_five[self.vocab[i]]=prob_neg
            if prob_pos>min(top_five.values()):
                key = GetKey(min(top_five.values()),top_five)
                top_five.pop(key)
                
                top_five[self.vocab[i]]=prob_pos
        
        top_five=sorted(top_five.items(),key=lambda x:x[1], reverse =True)
        bot_five=sorted(bot_five.items(),key=lambda x:x[1], reverse =True)

        print(top_five)
        print(bot_five)
            
        
    def predict(self,test_df):

        def total_probability(row):
            theta_pos=pd.Series(self.theta[0])
            theta_neg=pd.Series(self.theta[1])
            row=pd.Series(row[1:-1])
            row.index=[x for x in range(len(theta_pos))]
            temp=pd.DataFrame()
            temp['theta_pos']=theta_pos
            temp['theta_neg']=theta_neg
            temp['row']=row
            pos_predict=np.prod(temp['theta_pos'].loc[temp['row']!=0]*temp['row'].loc[temp['row']!=0])
            neg_predict=np.prod(temp['theta_neg'].loc[temp['row']!=0]*temp
            ['row'].loc[temp['row']!=0])
            if pos_predict==0 or neg_predict==0:
                print("ZERO")
                exit(0)
            likelihood_pos=round(pos_predict/(pos_predict+neg_predict),2)

            return pos_predict, neg_predict, likelihood_pos
            
        return zip(*test_df.apply(total_probability,axis=1))

    