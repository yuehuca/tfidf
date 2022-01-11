# -*- coding: utf-8 -*-

import os
import re
import time
import random
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def cleantxt(raw):
	fil = re.compile(u'[^0-9a-zA-Z.,~`!@#$%^&*()-=_+?;:<>/{}\[\]\\|""''\\n]+', re.UNICODE)
    
	return fil.sub(' ', raw)

def removenonen(data):
    data_en = data.copy()
    data_en["Review"] = np.nan
    for i_data in range(data.shape[0]):
        data_en.loc[i_data, "Review"] = cleantxt(data.loc[i_data, "Review"])
    
    return data_en

def encodetarget(data_en):
    data_en["Sentiment"] = np.nan
    for i_data_en in range(data_en.shape[0]):
        if data_en.loc[i_data_en, "RatingValue"] <= 2:
            data_en.loc[i_data_en, "Sentiment"] = 0
        elif data_en.loc[i_data_en, "RatingValue"] <= 3:
            data_en.loc[i_data_en, "Sentiment"] = 1
        elif data_en.loc[i_data_en, "RatingValue"] <= 5:
            data_en.loc[i_data_en, "Sentiment"] = 2
    
    return data_en

def checkbalance(df):
    for i_temp in range(3):
        print(df[df["Sentiment"]==i_temp].count()[0])
        print(df[df["Sentiment"]==i_temp].count()[0] / df.shape[0] * 100)

def balancesample(data_en, randomseed, cutoffratio):
    data_en_0 = data_en[data_en["Sentiment"]==0]
    data_en_1 = data_en[data_en["Sentiment"]==1]
    data_en_2 = data_en[data_en["Sentiment"]==2]
    
    np.random.seed(randomseed)

    data_en_2_shuffled = data_en_2.sample(frac=1, axis=0).reset_index(drop=True)
    data_en_2_cutoff = data_en_2_shuffled.loc[:(cutoffratio*data_en_2_shuffled.shape[0]), :]
    data_en_balanced = pd.concat([data_en_0, data_en_1, data_en_2_cutoff], axis=0)
    data_en_balanced = data_en_balanced.sample(frac=1, axis=0).reset_index(drop=True)
    
    return data_en_balanced

def buildpipline(Xy_train_original, Xy_valid_original, Xy_test, checker_testset, 
                 item_list_randomseed, item_list_cutoffratio, parameters):
    Xy_train = balancesample(Xy_train_original, item_list_randomseed, item_list_cutoffratio)
    
    X_train = Xy_train["Review"]
    y_train = Xy_train["Sentiment"]
    X_valid = Xy_valid_original["Review"]
    if checker_testset == 1:
        X_test = Xy_test["Review"]
    
    #GridCV pipline
    SGD_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                              alpha=1e-3, random_state=item_list_randomseed,
                                              max_iter=5, tol=None))])

    gs_clf = GridSearchCV(SGD_clf, parameters, cv=5, n_jobs=-1)
    gs_clf = gs_clf.fit(X_train, y_train)
    
    y_pred = gs_clf.predict(X_valid)
    if checker_testset == 1:
        y_test_pred = gs_clf.predict(X_test)
    
    df_y_pred = pd.DataFrame(y_pred)
    df_y_pred.columns = ["Pred_" + str(item_list_randomseed) + "_" + str(item_list_cutoffratio)]
    df_y_pred = df_y_pred.reset_index(drop=True)
    
    if checker_testset == 1:
        df_y_test_pred = pd.DataFrame(y_test_pred)
        df_y_test_pred.columns = ["Pred_" + str(item_list_randomseed) + "_" + str(item_list_cutoffratio)]
        df_y_test_pred = df_y_test_pred.reset_index(drop=True)
    else:
        df_y_test_pred = pd.DataFrame()
    
    return [df_y_pred, df_y_test_pred]

def resultlistdf(list_y_pred):
    df_list_y_pred = pd.concat(list_y_pred, axis=1)
    df_list_y_pred["y_pred_mode"] = np.nan
    for i_df_list_y_pred in range(df_list_y_pred.shape[0]):
        df_list_y_pred.loc[i_df_list_y_pred, "y_pred_mode"] = stats.mode(
            df_list_y_pred.iloc[i_df_list_y_pred, (df_list_y_pred.shape[1] - 2)])[0]
    
    return df_list_y_pred

def collectresults(Xy_train_original, Xy_valid_original, Xy_test, checker_testset, parameters, list_randomseed, list_cutoffratio):
    
    time_tick_difference = 0
    list_time_tick_difference = [0, 0, 0]
    looplength = len(list_randomseed) * len(list_cutoffratio)
    
    list_y_pred = []
    list_y_test_pred = []
    i_list_randomseed = 1
    i_totalloop = 1
    for item_list_randomseed in list_randomseed:
        i_list_cutoffratio = 1
        for item_list_cutoffratio in list_cutoffratio:
            time_tick_start = time.time()
            list_time_tick_difference.append(time_tick_difference)
            mean_time_tick_difference = np.mean(list_time_tick_difference[-3 : len(list_time_tick_difference)])
            time_left = mean_time_tick_difference * (looplength - i_totalloop)
            
            if i_totalloop > 3:
                print("...Randomseed: " + str(item_list_randomseed) + ", "
                      + str(i_list_randomseed) + " / " + str(len(list_randomseed))
                      + ", Cutoffratio: " + str(item_list_cutoffratio) + ", "
                      + str(i_list_cutoffratio) + " / " + str(len(list_cutoffratio))
                      + ", ETA: " + str(datetime.timedelta(seconds=time_left)) + ".")
            else:
                print("...Randomseed: " + str(item_list_randomseed) + ", "
                  + str(i_list_randomseed) + " / " + str(len(list_randomseed))
                  + ", Cutoffratio: " + str(item_list_cutoffratio) + ", "
                  + str(i_list_cutoffratio) + " / " + str(len(list_cutoffratio)) 
                  + ", ETA: " + str("...estimating."))
    
            temp_result = buildpipline(Xy_train_original, Xy_valid_original, Xy_test, checker_testset, 
                         item_list_randomseed, item_list_cutoffratio, parameters)
            
            list_y_pred.append(temp_result[0])
            list_y_test_pred.append(temp_result[1])
            
            i_list_cutoffratio += 1
            i_totalloop += 1
            
            time_tick_end = time.time()
            time_tick_difference = time_tick_end - time_tick_start
            
        i_list_randomseed += 1
    
        df_list_y_pred = resultlistdf(list_y_pred)
        
        if checker_testset == 1:
            df_list_y_test_pred = resultlistdf(list_y_test_pred)
        else:
            df_list_y_test_pred = pd.DataFrame()
    
    return [df_list_y_pred, df_list_y_test_pred]

def fixspecialword(Xy_valid_original, y_pred):
    df_Xy_valid = pd.DataFrame(Xy_valid_original)
    df_Xy_valid = df_Xy_valid.reset_index(drop=False)
    df_y_pred = pd.DataFrame(y_pred)
    df_y_pred.columns = ["Pred"]
    df_y_pred = df_y_pred.reset_index(drop=False)
    
    Xy_valid_pred = pd.concat([df_Xy_valid, df_y_pred], axis=1)
    
    list_special_words = ["terrible", "abysmal", "disappointment", "deteriorated",
                          "mouse", "mice", "rat", "cockroach", "roach", "unprofessional"]
    
    for i_Xy_valid_pred in range(Xy_valid_pred.shape[0]):
        for item_list_special_words in list_special_words:
            if re.match(str(".*" + item_list_special_words + ".*"), re.sub("\n","", 
                            (Xy_valid_pred.loc[i_Xy_valid_pred, "Review"]).lower())):
                if Xy_valid_pred.loc[i_Xy_valid_pred, "Pred"] != 0:
                    Xy_valid_pred.loc[i_Xy_valid_pred, "Pred"] = 0
    
    y_pred = Xy_valid_pred["Pred"].copy()
    
    return y_pred

##############################################################################

data = pd.read_csv('reviews.csv', delimiter='\t')
if os.path.exists('test.csv'):
    data_test = pd.read_csv('test.csv')
    #data_test = pd.read_csv('test.csv', delimiter='\t')
    checker_testset = 1
else:
    checker_testset = 0
    
#list_randomseed = [i for i in range(51)]
list_randomseed = random.sample(range(0, 100000), 3)
list_cutoffratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
validationsize = 0.3
parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)}

data_en = removenonen(data)
data_en = encodetarget(data_en)

Xy = pd.concat([data_en["Review"], data_en["Sentiment"]], axis=1)
Xy_train_original, Xy_valid_original = train_test_split(Xy, test_size=validationsize, random_state=list_randomseed[0])

Xy_train_original.to_csv("train.csv")
Xy_valid_original.to_csv("valid.csv")

Xy_train_original = pd.read_csv('train.csv').set_index('Unnamed: 0')
Xy_valid_original =  pd.read_csv('valid.csv').set_index('Unnamed: 0')

if checker_testset == 1:
    data_test_en = removenonen(data_test)
    #data_test_en = encodetarget(data_test_en)
    Xy_test = pd.concat([data_test_en["Review"], data_test_en["Sentiment"]], axis=1)
else:
    Xy_test = pd.DataFrame()

[df_list_y_pred, df_list_y_test_pred] = collectresults(Xy_train_original, Xy_valid_original, Xy_test, 
                                                       checker_testset, parameters, list_randomseed, list_cutoffratio)

y_pred = df_list_y_pred["y_pred_mode"]
y_valid = Xy_valid_original["Sentiment"]

#keyword fixing works too bad
#y_pred = fixspecialword(Xy_valid_original, y_pred)

print("Accuracy of Validation Set: ")
print(accuracy_score(y_valid, y_pred))
print("Performance Metrics of Validation Set: ")
print(metrics.classification_report(y_valid, y_pred, target_names=["negative", "neutral", "positive"]))

cm = confusion_matrix(y_valid, y_pred)
ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"]).plot()

if checker_testset == 1:
    y_test_pred = df_list_y_test_pred["y_pred_mode"]
    y_test = Xy_test["Sentiment"]
    
    #keyword fixing works too bad
    #y_test_pred = fixspecialword(Xy_test, y_test_pred)
    
    print("Accuracy of Test Set: ")
    print(accuracy_score(y_test, y_test_pred))
    print("Performance Metrics of Test Set: ")
    print(metrics.classification_report(y_test, y_test_pred, target_names=["negative", "neutral", "positive"]))
    
    cm_test = confusion_matrix(y_test, y_test_pred)
    ConfusionMatrixDisplay(cm_test, display_labels=["negative", "neutral", "positive"]).plot()




