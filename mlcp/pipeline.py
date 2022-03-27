from . import classifires as cl
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn.model_selection import cross_validate
from sklearn import linear_model
from sklearn import feature_selection as fs
import sklearn.decomposition as skde
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import math
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
LE = LabelEncoder()
OHE = OneHotEncoder(handle_unknown='ignore')
import joblib
import ast
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_error
import math
import os
cpath = os.path.abspath(os.getcwd())


#------read & view-----
def read_data(filepath):
    df = pd.read_csv(filepath)
    return df


def missing_value_analysis(df,drop_th):
    samples = len(df); dropped_col=[]
    dfc=df.copy(); dfc=dfc.dropna()
    missing_dict = {}
    for c in df.columns:
        n_missing = df[c].isnull().sum()
        missing_ratio = n_missing/samples
#        x_scores=feature_scores(dfc.drop([c],axis=1), dfc[c]
        
        if missing_ratio >= drop_th:
            df = df.drop([c], axis=1)
            dropped_col.append(c)
        elif n_missing > 0:
            missing_dict[c] = []
            missing_dict[c].append((n_missing, missing_ratio, df[c].dtype))
            
    print("dropped_columns-->", dropped_col)
    print("other missing colums-->")
    for k,v in missing_dict.items():
        print(k, "=" , v); print("")
    return df
    

def correlations(x, th=0.5):
   cors = x.corr()
   return cors[abs(cors)>th]



def bias_analysis(df,y_name):
    cv_y = df[y_name].value_counts()
    min_y = min(cv_y)/len(df); max_y = max(cv_y)/len(df)
    sample_diff = max_y-min_y;
    return sample_diff, min_y, max_y



def visualize_y_vs_x(df,y_name):
    ds=[];i=0; num_cols=[]
    for c in sorted(df[y_name].unique()):
        df_name =  "df"+str(c)
        df_name = df[df[y_name]==c]
        ds.append([])
        for s in df.columns:
            if df[s].dtype != 'O':
                ds[i].append(list(df_name[s]))
                ds[i].append(list(df_name[y_name]))
                num_cols.append(s)
        i=i+1
        
    ds = np.array(ds); print(ds.shape)
    
    i=0
    colors = ['blue','green','yellow','red','black','orange']
    for j in range(0,len(ds[0]),2):
        print(num_cols[i]); i=i+1
        for k in range(len(df[y_name].unique())):
            plt.scatter(ds[k][j],ds[k][j+1],c=colors[k])
        plt.show()


#--------feature analysis----------#
def sort_by_value(dictx):
    dicty=sorted(dictx.items(),key=lambda item: item[1],reverse=True)
    return dicty


def data_to_df(x):
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x)
    return x


def is_categorical(data):
    uv = len(list(set(data))); #print("unique-->", uv)
    if (uv/len(data) <= 0.05 or uv <= 10) and type(data[0]) != float:
        return 1
    else:
        return 0


def assert_to_discrete(data):
    data = np.array(data)
    if max(data)<10:
        try:
            data = data*10;
        except:
            pass
    if is_categorical(data) == 1:
        cat_data = data.astype('str')
    else:
        s = int(min(data))
        e = int(max(data))
        itr = int(np.std(data))
        bins = []; #print("itr-------------->",itr)
        
        for i in range(s,e,itr):
            bins.append(i)
            
        bins.append(e + itr)
        bins.append(bins[-1] + itr); #buffer

        bin_range = {}
        for i in range(len(bins)-1):
            bin_range[i]=[bins[i],bins[i+1]]

        cat_data=[]; missed=[]
        for v in data:
            f=0
            for i, br in bin_range.items():
                if v >= br[0] and v < br[1]:
                    f=1
                    # print("c1", v, br, "--->", i)
                    cat_data.append(i); break
            if f==0: missed.append(v)

    # print("bin range---> ", s, e, itr)
    # print("bins--->", bins)
    # print("bin dict--->", bin_range)
    # print("missed to bin--->", missed)
    # print('counter--->', token_counter(cat_data))
    return cat_data


def feature_analysis(x,y):
    dfx = data_to_df(x).copy()
    #by default add 'ybin' as categorical
    if y.dtype != 'O':
        dfx['ybin'] = assert_to_discrete(y)
    else:
        dfx['ybin'] = y

    label_count = len(dfx['ybin'].unique());
    # print('label-->', label_count)

    feature_scores={}
    for col in dfx.columns:
#        print("feature analysis--->", col)
        if col == 'ybin': continue
        x_type = dfx[col].dtype
        
        if x_type != 'O':
            if np.max(dfx[col])<1:
                dfx[col] = dfx[col] + 1
            #print(len(x[col]))
            try:
                discrete = assert_to_discrete(dfx[col]); #print(len(discrete))
                dfx[col] = discrete
            except:
                pass

        gx = dfx.groupby([col,'ybin'])['ybin'].count(); #print(gx)
        gxr = gx/len(dfx)
        fgx = gxr[gxr.values >= 0]; #print(fgx)
        ix = fgx.index
        clist={}
        for c, yb in ix:
            if c not in clist: clist[c] = []
            clist[c].append(yb)
        availability = len(clist)/len(dfx[col].unique())
        penalty=[]
        for k, v in clist.items():
            penalty.append(1/len(v))
        avg_penalty = np.mean(np.array(penalty))
        ns = availability*avg_penalty
        feature_scores[col]=ns
    feature_scores = sort_by_value(feature_scores)
    return feature_scores



#-----feature engineering + selection + reduction------

def feature_selection(X,Y, model):
    X, Y, mmd = auto_transform_data(X,Y)
    feature_folds = ms.KFold(n_splits=10, shuffle = True)
    #customize parameter if needed
#    logistic_mod = linear_model.LogisticRegression(C = 10)
    selector = fs.RFECV(estimator = model, cv = feature_folds,
                      scoring = 'roc_auc')
    selector = selector.fit(X, Y)
    r = selector.ranking_
    c = X.columns
    ranks={}
    for i in range(len(r)):
        ranks[c[i]]=r[i]
    return ranks
    


def fscore(a,b):
    return (a*b)/(a+b)


def combine(a,b):
    return a.astype(str)+b.astype(str)


def euclidean_distance(xdims):
    #sqrt((x1-x2)^2 + (y1-y2)^2)
    dist=[]; diff_sum=[]
    r=0
    for data in xdims:
        diff_sum.append([])
        for dim in data:
            diff=''
            for i in range(len(dim)):
                if diff == '':
                    diff = dim[i]
                else:
                    diff = diff-dim[i]
            diff_sum[r].append((diff)**2)
        dist.append(math.sqrt(np.sum(np.array(diff_sum[r]))))
        r=r+1
    return dist



def polynomial_features(x,degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    i=0
    for c in x.columns:
        if x[c].dtype != 'O':
            i=i+1
            poly_feat = poly.fit_transform(x[c].reshape(-1,1)); #print(poly_feat[0])
            poly_df = pd.DataFrame(data=poly_feat, columns = ["f"+str(i)+str(d) for d in range(degree)])
            x = x.drop([c],axis=1)
            x=x.join(poly_df)
    return x


class GaussianFeatures(BaseEstimator, TransformerMixin):
    """Uniformly spaced Gaussian features for one-dimensional input"""
    
    def __init__(self, N, width_factor=2.0):
        self.N = N
        self.width_factor = width_factor
    
    @staticmethod
    def _gauss_basis(x, y, width, axis=None):
        arg = (x - y) / width
        return np.exp(-0.5 * np.sum(arg ** 2, axis))
        
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        return self._gauss_basis(X[:, :, np.newaxis], self.centers_,
                                 self.width_, axis=1)


def gaussian_features(x,y,n_dim):
    i=0
    for c in x.columns:
        if x[c].dtype != 'O':
            i=i+1
            gaus_fit = GaussianFeatures(n_dim).fit(x[c],y)
            gaus_feat = gaus_fit.transform(x[c].reshape(-1,1))
            gaus_df = pd.DataFrame(data=gaus_feat, columns = ["f"+str(i)+str(d) for d in range(n_dim)])
            x = x.drop([c],axis=1)
            x=x.join(gaus_df,lsuffix='_left', rsuffix='_right')
    return x


def oversampling(x,y):
    smote = SMOTE()
    x,y = smote.fit_resample(x,y)
    return x,y


def matrix_correction(x):
    x = pd.DataFrame(x)
    for c in x.columns:
        x[c] = x[c].replace(to_replace=np.nan, value=0)
        x[c] = x[c].replace(to_replace=np.inf, value=sorted(list(x[c]),reverse=True)[1])
        x[c] = x[c].replace(to_replace=-np.inf, value=sorted(list(x[c]),reverse=True)[-2])
    return x.values


#-----transformations-------
    
def skew_correction(X):
    for col in X.columns:
        if X[col].dtype != 'O':
            skewval = X[col].skew()
            if abs(skewval) > 0.7:
                X[col] = np.log1p(X[col])
    return X


def max_normalization(X):
    feature_max=[]
    for col in X.columns:
        if X[col].dtype != 'O':
            fmax = max(X[col])
            X[col] = X[col]/fmax
            feature_max.append(fmax)
    with open(cpath+"/models/feature_max.txt","w") as f:
        f.write(str(feature_max))
    return X, feature_max


def minmax_normalization(X):
    scaler = preprocessing.MinMaxScaler().fit(X)
    X = scaler.transform(X)
    joblib.dump(scaler,cpath+"/models/norm_scaler.pkl")
    return X


def Standardization(X):
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    joblib.dump(scaler,cpath+"/models/std_scaler.pkl")
    return X


def split_num_cat(X):
    x_num = X.select_dtypes(exclude=['O'])
    x_cat = X.select_dtypes(include=['O'])
    return x_num, x_cat


def join_num_cat(x_num, x_cat):
    X = np.concatenate((x_num,x_cat),axis=1)
    X = pd.DataFrame(X)
    return X


def label_encode(X):
    try:
        d = X.shape[1]
        for col in X.columns:
            labels = LE.fit_transform(X[col].values)
            X[col] = labels.astype(str)
    except:
        X = LE.fit_transform(X.values)
        
    return X


def onehot_encode(X_cat):
    print(X_cat.shape)
    X_cat_le = X_cat.apply(lambda col: LE.fit_transform(col)); #print(S)
    OHE_model = OHE.fit(X_cat_le)
    joblib.dump(OHE_model,cpath+"/models/ohe_model.pkl")
    X_cat_ohe = OHE.transform(X_cat_le).toarray();  #print(2,F.shape, F[0])
    return X_cat_ohe
            

def auto_transform_data(X, Y):
    #log transform, one-hot encoding, vectorization done - yet to normalize
    NewX, NewY, model_meta_data = cl.prepare_data(X, Y)
    return NewX, NewY, model_meta_data


def reduce_dimensions(X, n_dim):
    pca_model = skde.PCA(n_components=n_dim)
    pca_fit = pca_model.fit(X)
    X = pca_fit.transform(X)
    print("PCA variance:")
    print(pca_fit.explained_variance_ratio_); print("")
    joblib.dump(pca_fit, cpath+"/models/pca_model.pkl")
    return X

#-----validations-----

def stratified_sample(df, col, n_samples):
    n = min(n_samples, df[col].value_counts().min())
    df_ = df.groupby(col).apply(lambda x: x.sample(n))
    df_.index = df_.index.droplevel(0)
    return df_


def compare_models(x, y,rstate):
    mmd = cl.compare_models(x,y,rstate)
    for k, v in mmd.items():
        print(k, ":", v)
    return mmd
    

def kfold_cross_validate(model,X,Y,rstate):
    feature_folds = ms.KFold(n_splits=5, shuffle = True, random_state=rstate)
    cv_estimate = ms.cross_val_score(model, X, Y, cv = feature_folds)
    print('Mean performance metric = ', np.mean(cv_estimate))
    print('SDT of the metric       = ', np.std(cv_estimate))
    print('Outcomes by cv fold')
    for i, x in enumerate(cv_estimate):
        print('Fold ', (i+1, x))
    print("")
  

def select_best_parameters(model, param_grid, X, Y,rstate):
    feature_folds = ms.KFold(n_splits=5, shuffle = True,random_state=rstate)
    clf = ms.GridSearchCV(estimator = model, 
                          param_grid = param_grid, 
                          cv = feature_folds,
                          scoring = 'roc_auc',
                          return_train_score = True)
    clf.fit(X, Y)
    print(clf.best_estimator_, clf.best_score_)
    return clf.best_estimator_



def threshold_prediction(model, x_test, y, pred_th):
    test_pred_prob = model.predict_proba(x_test); #print(test_pred_prob[0], test_pred[0])
    test_pred = []
    for p in test_pred_prob:
        th_met = 0
        for v in sorted(list(y.value_counts().index)):
            if p[v]>pred_th:
                test_pred.append(v)
                th_met=1
                break
        if th_met==0:
            test_pred.append(-1)
    test_pred = np.array(test_pred)
    return test_pred



def clf_train_test(model, x, y, rstate, model_id,pred_th=None):
    print(model); print("")
    x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.2, random_state=rstate)
    model_fit = model.fit(x_train,y_train)
    joblib.dump(model_fit, cpath+"/models/model_"+str(model_id))
    train_pred = model_fit.predict(x_train)
    
    if pred_th != None:
        test_pred = threshold_prediction(model_fit, x_test, y, pred_th)
    else:
        test_pred = model_fit.predict(x_test)
       
    print("Training:")
    print(classification_report(y_train, train_pred))
#    print("roc_auc:", roc_auc_score(y_train, train_pred))
    print("")
    print("% of Unknown classe @ threshold = "+str(pred_th), " is ", round(len(test_pred[test_pred==-1])/len(test_pred),3))
    print("Testing:")
    print(classification_report(y_test, test_pred))
#    print("roc_auc:", roc_auc_score(y_test, test_pred))
    return model_fit
    
    
    
    
def reg_train_test(model, x, y, rstate, model_id):
    print(model); print("")
    x_train,x_test,y_train,y_test=ms.train_test_split(x,y,test_size=0.3, random_state=rstate)
    model_fit = model.fit(x_train, y_train)
    joblib.dump(model_fit, cpath+"/models/model_"+str(model_id))
    y_pred=model.predict(x_test)
#    y_test=np.expm1(y_test)
#    y_pred=np.expm1(y_pred)
    plt.scatter(y_test,y_pred)
    plt.grid()
    plt.xlabel('Actual y')
    plt.ylabel('Predicted y')
    plt.title('actual y vs predicted y')
    plt.show()
    rmse=math.sqrt(mean_squared_error(y_test, y_pred)); print("RMSE = ", rmse)
#    den = np.expm1(y)
    den = y
    er=rmse/np.mean(den); print("Error Rate (RMSE/Y_mean) = ", round(er*100,2),"%")
    eth = rmse*0.1
    diff = (y_test-y_pred).values
    abs_diff = abs(diff)
    best = diff[abs_diff<=eth]
    bp=len(best)/len(abs_diff); print("Prediction Rate (Predictions with <=10% of RMSE) = ", round(bp*100,2),"%")
    o_pred = len(diff[diff<0]); print("Over-Prediction % = ", round((o_pred/len(diff))*100,2),"%")
    u_pred = len(diff[diff>0]); print("Under-Prediction % = ", round((u_pred/len(diff))*100,2),"%")

    