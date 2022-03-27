import joblib, sklearn
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
LE = LabelEncoder()
OHE = OneHotEncoder(handle_unknown='ignore')
from sklearn import model_selection
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import StackingClassifier
######----default name for label is "class"--------#####

def sort_by_value(dictx):
    dicty=sorted(dictx.items(),key=lambda item: item[1],reverse=True)
    return dicty


def data_to_df(x):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return x


def tex2vec(X):
    vectorizer = TfidfVectorizer(min_df=0.001, max_df=1.0, stop_words='english')
    X = X.values.astype('U')
    xlist = [t.lower() for t in X]
    vectors = vectorizer.fit(xlist)
    vectors = vectorizer.transform(xlist)
    vectors = vectors.toarray()
    return vectors, vectorizer
    

 
def get_models():
    models = {}
    models['LogR'] = LogisticRegression()
    models['KNN'] = KNeighborsClassifier()
    models['DTC'] = DecisionTreeClassifier()
    models['NBC'] = MultinomialNB()
    models['SVC'] = SVC()
    models['RFC'] = RandomForestClassifier()
    models['GBC'] = GradientBoostingClassifier()
#    models['XGB'] = XGBClassifier()
    models['MLP'] = MLPClassifier()
    return models


def evaluate_model(model, X, y, rstate):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=rstate)
	scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1, error_score='raise')
	return scores


def get_stacking(models):
	 # define the base models
    level0 = list(); #base models
    for k, m in models.items():
        level0.append((k, m))
	 # define meta learner model
    level1 = models['LogR']; #meta model
	 # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


def prepare_data(X,Y):
    model_meta_data={'vectorized':[], 'one-hot_encoded':[], 'log_transformed':[]}
    n = 100
    X = data_to_df(X)
    NewX = np.array([[0] for i in range(len(X))]);  # print(NewX.shape)
    for col in X.columns:
        print(col)
        if X[col].dtype == 'O':
            sample = X[col][:n]
            tokens = [len(t.split()) for t in sample]
            avg_tokens = np.mean(np.array(tokens))
            cat_ratio = len(sample.unique()) / n
            if cat_ratio > 0.3 and avg_tokens >= 3:
                F, vec = tex2vec(X[col]);  # print(1,F.shape, F[0])
                NewX = np.concatenate((NewX, F), axis=1)
                model_meta_data['vectorized'].append(col)
            else:
#                print(set(list(X[col])))
                label = LE.fit_transform(list(X[col].values)).reshape(-1, 1); #print(S)
                OHE_model = OHE.fit(label)
                F = OHE.transform(label).toarray();  #print(2,F.shape, F[0])
                NewX = np.concatenate((NewX, F), axis=1)
                model_meta_data['one-hot_encoded'].append(col)
        else:
            if abs(X[col].skew()) > 0.7:
                X[col] = np.log1p(X[col])
                model_meta_data['log_transformed'].append(col)
            F = X[col].values.reshape(-1, 1);  # print(3,F.shape, F[0])
            NewX = np.concatenate((NewX, F), axis=1)

    # As we need to clean X, add Y and clean to maintain the dimensions
    LE.fit(list(Y))
    NewY = LE.transform(Y)
    NewX = data_to_df(NewX)
    NewX = NewX.iloc[0:, 1:];  # print(NewX.shape)
    return NewX, NewY, model_meta_data



def compare_models(X,Y,rstate):
    model_meta_data = {}
    #get models
    models = get_models()
    #add staking as model
    models['Stacking'] = get_stacking(models)

    # evaluate each model in turn
    temp = {}
    model_list = {}
    for name, model in models.items():
        scores = evaluate_model(model, X, Y, rstate)
        cvmean = round(mean(scores), 4)
        cvstd = round(std(scores), 4)
        model_list[name] = {'mean': cvmean, 'std': cvstd}
        temp[name] = cvmean
        print(name, ':', cvmean, " && ", cvstd); print("")

    final_model = sort_by_value(temp)[0]
    model_meta_data['best_model'] = final_model
    model_meta_data['models_data'] = model_list
    return model_meta_data


