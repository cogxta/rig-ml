import mlcp.pipeline as pl
import mlcp.classifires as cl
import mlcp.regressors as rg
import numpy as np
from datetime import datetime as dt
import math
import matplotlib.pyplot as plt
from mlcp.ann import ann_train_test

import warnings
warnings.filterwarnings("ignore")
#execution controls
classification=1; #if regression = 0, classification = 1
read=1
primary_analysis=0 #dev only
visual_analysis=0
observed_corrections=1
analyze_missing_values=0
treat_missing_values=0
define_variables=1
analyze_stats=1; #dev only
analyzed_corrections=0;
oversample=0; #dev only
feature_engineering=0
gaussian_transform=0
polynomial_transform=0
skew_corrections=0
scaling=1
encoding=0
matrix_corrections=0
reduce_dim=0
compare=0; #dev only
cross_validate=0; #dev only
grid_search=0; #dev only
train_classification=1
train_regression=0


if read==1:
    filepath = "data/GBC_3/cancer.csv"
    y_name = 'diagnosis'
    dtype_file = "project3_dtype_analysis.txt"
    df = pl.read_data(filepath)
    if classification == 1:
        cv_y = df[y_name].value_counts()
        min_y = min(cv_y)/len(df); max_y = max(cv_y)/len(df)
        sample_diff = max_y-min_y; print("sample diff:", sample_diff)


if primary_analysis==1:
    #consider: unwanted features, numerical conversions (year to no. years), 
    #wrong dtypes, missing values
    df_h = df.head()
    with open(dtype_file, "w") as f:
        for c in df_h:
            line1 = df_h[c]
            line2 = df[c].nunique()
            line3 = df[c].isnull().sum()
            f.write(str(line1) + "\n" + "Unique: " + str(line2) + 
                    ", missing: " + str(line3)
            + "\n\n" + "-----------------"+"\n")
    if classification == 0:
        plt.boxplot(df[y_name]); plt.show()



if visual_analysis==1:
    ds=[];i=0
    for c in sorted(df[y_name].unique()):
        df_name =  "df"+str(c)
        df_name = df[df[y_name]==c]
        ds.append([])
        for s in df.columns:
            ds[i].append(list(df_name[s]))
            ds[i].append(list(df_name[y_name]))
        i=i+1
        
    ds = np.array(ds); print(ds.shape)
    
    i=0
    colors = ['blue','green','yellow','red','black','orange']
    for j in range(0,len(ds[0]),2):
        print(df.columns[i]); i=i+1
        for k in range(len(df[y_name].unique())):
            plt.scatter(ds[k][j],ds[k][j+1],c=colors[k])
        plt.show()



if observed_corrections==1:
    df = df.drop(['id','fractal_dimension_worst','concavity_worst','texture_se'],axis=1)
    df[y_name] = pl.label_encode(df[y_name])
    print("y dtype---->", df[y_name].dtype)
    pass
    
    
if analyze_missing_values==1:
    drop_th = 0.4
    print(df.shape)
    df = pl.missing_value_analysis(df, drop_th)
    print(df.shape)
    before = len(df); df_copy_drop = df.dropna(); after = len(df_copy_drop); 
    print("dropped %--->", round(1-(after/before),2)*100,"%")
    num_df = df.select_dtypes(exclude=['O'])
    

if treat_missing_values==1:
    pass


if define_variables==1:
    y = df[y_name]
    x = df.drop([y_name],axis=1); #print(x.info())
    n_dim = x.shape[1]
    print(x.shape)
    if y.dtype== 'O':
        print(y.value_counts())
    else:
        print("y skew--->", y.skew())


if analyze_stats==1:
   #find important features and remove correlated features based on low-variance or High-skew
    cors = pl.correlations(x, th=0.7)
    with open(dtype_file, "a") as f:
        f.write("\n\n\n"+str(cors))
    scores = pl.feature_analysis(x,y); print(scores); print("")
#    if classification == 1:
#        ranks = pl.feature_selection(x,y); print(ranks); print("")
    print("skew in feature:")
    print(x.skew())
    
    
if analyzed_corrections==1:
    pass
    

if oversample==1:
    #for only imbalanced data
    x,y = pl.oversampling(x,y)
    print(x.shape); print(y.value_counts())
    

def fscore(a,b):
    return (a*b)/(a+b)

def combine(a,b):
    return a.astype(str)+b.astype(str)


if feature_engineering==1:
   #subjective and optional - True enables the execution
   print("Initial feature:");print(x.head(1));print("")

         
if polynomial_transform==1:
   degree=6
   x = pl.polynomial_features(x,degree)
   print("polynomial features:")
   print(x.head(1)); print("")


if gaussian_transform==1:
   n_dim=5
   x = pl.gaussian_features(x,y,n_dim)
   print("Gaussian features:")
   print(x.head(1)); print(x.shape,y.shape);print("")
   

if skew_corrections==1:
    x = pl.skew_correction(x)


if scaling==1:
    x_num, x_cat = pl.split_num_cat(x)
    
    if False:
        x_num, fm = pl.max_normalization(x_num); #0-1
    if True:
        x_num = pl.minmax_normalization(x_num) ; #0-1
    if False:
        x_num = pl.Standardization(x_num.values); #-1 to 1
    
    print("")
    print("after scaling - categorical-->", x_cat.info())     
    x = pl.join_num_cat(x_num,x_cat)


if encoding==1:
    x_num, x_cat = pl.split_num_cat(x)
    
    if True:
        x_cat = pl.label_encode(x_cat)
    if False:
        x_cat = pl.onehot_encode(x_cat)
    
    if False:
         x,y,mmd = pl.auto_transform_data(x,y); #best choice if dtypes are fixed
    
    x = pl.join_num_cat(x_num,x_cat)     
         

if matrix_corrections==1:   
    print("before matrix correction:", x.shape)
    x,y = pl.matrix_correction(x,y); 
    print("after matrix correction:", x.shape)  
    

if reduce_dim==1:
    x = pl.reduce_dimensions(x, 2); #print(x.shape)
    print("transformed x:")
    print(x.shape); print("")
    

if compare==1:
    model_meta_data = pl.compare_models(x, y, 111)
    best_model_id = model_meta_data['best_model'][0]
    best_model = cl.get_models()[best_model_id]; print(best_model)
    
    
if cross_validate==1:
    best_model = cl.LogisticRegression()
    pl.kfold_cross_validate(best_model, x, y,111)


if grid_search==1:
    #grids
    dtc_param_grid = {"criterion":["gini", "entropy"],
                      "class_weight":[{0:1,1:1}],
                      "max_depth":[2,4,6,8,10],
                      "min_samples_leaf":[1,2,3,4,5],
                      "min_samples_split":[2,3,4,5,6],
                      "random_state":[21,111]
                      }
    
    log_param_grid = {"penalty":['l1','l2','elasticnet'],
                      "C":[0.1,0.5,1,2,5,10],
                      "class_weight":[{0:1,1:1}],
                      "solver":['liblinear', 'sag', 'saga'],
                      "max_iter":[100,150,200,300],
                      "random_state":[21,111]
                      }
    
    param_grid = log_param_grid
    model = cl.LogisticRegression()
    best_param_model = pl.select_best_parameters(model, param_grid, x, y, 111)


if train_classification==1:
   print(x.shape)
   trained_model = ann_train_test(np.array(x,float),y.values,'keras','cancer')

    
if train_regression==1:
   model = rg.GradientBoostingRegressor()
   pl.reg_train_test(model,x,y,111,"GBR1")
