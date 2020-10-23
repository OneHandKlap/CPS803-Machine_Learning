import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn
import preprocessor 
import bayes_model

class Analyzer(object):

    def __init__(self,model,test_path):
        self.model=model
        self.test_path=test_path


    def threshold_scan(self,thresholds,output_path):
        metrics=pd.DataFrame()

        test_df=pd.read_csv(self.test_path,names=['x','y'])

        test_data=preprocessor.Preprocessor(test_df,self.model.vocab)
        test_data.tokenize('x')
        test_data.add_tags('x')
        test_data.lemmatize('x')

        test_data.update_dataframe('x','y')

        test_data.data['pos_score'],test_data.data['neg_score']=(self.model.predict(test_data.data))

        
        def make_judgement(row,threshold):
            
            if row['pos_score']/(row['pos_score']+row['neg_score'])>threshold:
                return 1
            else:
                return 0
        metric_acc=[]
        for i in thresholds:
            test_data.data['results']=test_data.data.apply(make_judgement, args=(i,),axis=1)
            test_data.data['results']=test_data.data['results'].astype('bool')
            test_data.data['y']=test_data.data['y'].astype('bool')
            count_true=sum(test_data.data['results'])
            count_false=sum(~test_data.data['results'])
            label_true=sum(test_data.data['y'])
            label_false=sum(~test_data.data['y'])
            true_pos=sum(test_data.data['results']&test_data.data['y'])
            true_neg=sum(~test_data.data['results']&~test_data.data['y'])
            false_pos=sum(test_data.data['results']&~test_data.data['y'])
            false_neg=sum(~test_data.data['results']&test_data.data['y'])
            accuracy=(true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg)
            try:
                precision=true_pos/(true_pos+false_pos)
            except ZeroDivisionError:
                precision= 0
                
            try:
                recall=true_pos/(true_pos+false_neg)
            except ZeroDivisionError:
                recall=0
            try:
                spec=true_neg/(true_neg+false_pos)
            except ZeroDivisionError:
                spec=0
            try:
                f1=(2*precision*recall)/(precision+recall)
            except ZeroDivisionError:
                f1=0

            metric_acc.append([i,true_pos,true_neg,false_pos,false_neg,accuracy,precision,recall,spec,f1])

        metrics=pd.DataFrame(metric_acc)

        
        metrics.columns=['threshold','tp','tn','fp','fn','accuracy','precision','recall','specificity','harmonic_mean']
        metrics.to_csv(output_path)
        
    #analyzes threshold scan csv and finds threshold value that gives best confusion matrix, then generates visual
    def confusion_matrix(self,threshold_scan):
        pd.read_csv(threshold_scan_path)
        best_matrix=None

