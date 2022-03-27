import mlcp.pipeline as pl
import mlcp.classifires as cl
import mlcp.regressors as rg
import numpy as np
from datetime import datetime as dt
import math

import warnings
warnings.filterwarnings("ignore")
#execution controls
classification=0; #if regression = 0, classification = 1
read=1
primary_analysis=0 #dev only
observed_corrections=1
analyze_missing_values=1
treat_missing_values=1
define=1
analyze_stats=0; #dev only
analyzed_corrections=1;
oversample=0; #dev only
feature_engineering=1
skew_corrections=1
transform=1
matrix_corrections=0
reduce_dim=0
compare=0; #dev only
cross_validate=0; #dev only
grid_search=0; #dev only
train_classification=0
train_regression=1


if read==1:
    #consider: unwanted features, numerical conversions (year to no. years), 
    #wrong dtypes, missing values
    filepath = "data/LIN_1/HousePrices.csv"
    y_name = 'Property_Sale_Price'
    df = pl.read_data(filepath)


if primary_analysis==1:
    df_h = df.head()
    with open("project1_dtype_analysis.txt", "w") as f:
        for c in df_h:
            line1 = df_h[c]
            line2 = df[c].nunique()
            line3 = df[c].isnull().sum()
            f.write(str(line1) + "\n" + "Unique: " + str(line2) + 
                    ", missing: " + str(line3)
            + "\n\n" + "-----------------"+"\n")
    if classification == 0:
        plt.boxplot(df[y_name]); plt.show()


if observed_corrections==1:
    df = df.drop(['Id', 'MoSold', 'YrSold'],axis=1)
    
    to_no_years = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
    cur_yr = dt.today().year
    for c in to_no_years:
        df[c] = cur_yr - df[c]; print(df[c].head())

    num_to_object = ['Dwell_Type']
    for c in num_to_object:
        df[c] = df[c].astype(str)
    
    rating_dict = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
    ordinals=[]
    for c in df.columns:
       uv = list(df[c].value_counts().index)
       assign = 1
       for u in uv:
          if u not in rating_dict: assign=0; break
       
       if assign==1:
           for k, v in rating_dict.items():
               if c not in ordinals: ordinals.append(c)
               df[c][df[c]==k] = v
         
               
    print("ordinal columns--->", ordinals)
    for oc in ordinals:
        print(oc)
        df[oc] = df[oc].fillna(0)
        df[oc] = df[oc].astype(int)
    
    #outlier data in Y
#    df = df[df[y_name]<400000].reset_index()
#    df[y_name] = np.log1p(df[y_name])
    pass
    
    
if analyze_missing_values==1:
    drop_th = 0.4
    print(df.shape)
    df = pl.missing_value_analysis(df, drop_th)
    print(df.shape)
    before = len(df); df_copy_drop = df.dropna(); after = len(df_copy_drop); 
    print("dropped %--->", round(1-(after/before),2)*100,"%")
    num_df = df.select_dtypes(exclude=['O'])
    pl.correlations(num_df, th=0.5)
    

if treat_missing_values==1:
    fill_with_mean_mode = ['LotFrontage', 'MasVnrType','MasVnrArea', 'BsmtQual', 
                           'BsmtCond', 'BsmtExposure', 
                           'BsmtFinType1', 'BsmtFinType2', 'Electrical']
    
    for c in fill_with_mean_mode:
        if df[c].dtype != 'O':
            df[c] = df[c].fillna(np.mean(df[c]))
        else:
            df[c] = df[c].fillna(df[c].value_counts().index[0])
    
    missing_not_random = ['GarageType', 'GarageYrBlt', 'GarageFinish']
    #Garage is missing is not at random, the respectives houes are not having garage.
    # so we need to phenalize if there is numercial feature.
    for c in missing_not_random:
        if df[c].dtype != 'O':
            df[c] = df[c].fillna(-1)
        else:
            df[c] = df[c].fillna('No Garage')
    
    print(df.info())
    

if define==1:
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
    pl.correlations(x, th=0.7)
    scores = pl.feature_analysis(x,y); print(scores); print("")
    if classification == 1:
        ranks = pl.feature_selection(x,y); print(ranks); print("")
    print(x.skew())
    
    
if analyzed_corrections==1:
    #multicollinear - GarageArea & GarageCars - correlating 88%
#    least_features = [f for f,s in scores[-5:]]; print(least_features)
    pass
    

if oversample==1:
    #for only imbalanced data
    x,y = pl.oversampling(x,y)
    print(x.shape); print(y.value_counts())
    

if feature_engineering==1:
   #subjective and optional - True enables the execution
   print("Initial feature:");print(x.head(1));print("")
      
   df['age'] = (df['YearBuilt']+df['YearRemodAdd'])/2
   
   df = df.drop(['YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                          ], axis=1)
   
   if False:
       degree=3
       x = pl.polynomial_features(x,degree)
       print("polynomial features:")
       print(x.head(1)); print("")
    
   if False:
       n_dim=10
       x = pl.gaussian_features(x,y,n_dim)
       print("Gaussian features:")
       print(x.head(1)); print(x.shape,y.shape);print("")
   

if skew_corrections==1:
    x = pl.skew_correction(x)


if transform==1:
    x_num, x_cat = pl.split_num_cat(x)
    
    if False:
        x_num, fm = pl.max_normalization(x_num)
    if False:
        x_num = pl.minmax_normalization(x_num)
    if True:
        x_num = pl.Standardization(x_num.values)
        
    x = pl.join_num_cat(x_num,x_cat)
        
#    print(x.shape, type(x))
    if True:
        x = pl.label_encode(x)
    if False:
        x = pl.onehot_encode(x)
    
    if False:
         x,y,mmd = pl.auto_transform_data(x,y); #best choice if dtypes are fixed


if matrix_corrections==1:   
    print("before matrix correction:", x.shape)
    x,y = pl.matrix_correction(x,y); 
    print("after matrix correction:", x.shape)  
    

if reduce_dim==1:
    x = pl.reduce_dimensions(x, 1000); #print(x.shape)
    print("transformed x:")
    print(x.shape); print("")
    

if compare==1:
    model_meta_data, Newx = pl.compare_models(x, y)
    best_model_id = model_meta_data['best_model'][0]
    best_model = cl.get_models()[best_model_id]; print(best_model)
    
if cross_validate==1:
    best_model = cl.LogisticRegression()
    pl.kfold_cross_validate(best_model, x, y,111)

if grid_search==1:
    best_model = cl.LogisticRegression()
    
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
                      "solver":['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      "max_iter":[100,150,200,300],
                      "random_state":[21,111]
                      }
    
    param_grid = log_param_grid
    best_param_model = pl.select_best_parameters(best_model, param_grid, x, y, 111)


if train_classification==1:
    best_param_model = cl.LogisticRegression(C=1, class_weight={0: 1, 1: 1}, penalty='l1',
                   random_state=21, solver='liblinear')
    pl.train_test(best_param_model,x,y,111,"DTC1")
  
    
if train_regression==1:
   model = rg.GradientBoostingRegressor(random_state=21)
   pl.reg_train_test(model,x,y,111,"GBR1")
    