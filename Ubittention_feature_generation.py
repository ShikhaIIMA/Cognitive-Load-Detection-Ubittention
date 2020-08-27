
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import librosa
from scipy.stats import kurtosis 
import scipy.stats
import xgboost as xgb

#defining functions for extracting features
def get_autocorr(df):
    autocorr_feature=[]
    df=pd.DataFrame(df)
    for col_id in range(0,df.shape[0]):
       curr_col=pd.Series(df.loc[col_id,:])
       autocorr_feature.append(curr_col.autocorr(lag=1))
    return autocorr_feature;  

def get_iqr(df):
    iqr_feature=[]
    df=pd.DataFrame(df)
    for col_id in range(0,df.shape[0]):
       curr_col=pd.Series(df.loc[col_id,:])
       iqr_feature.append(scipy.stats.iqr(curr_col))
    return iqr_feature;  


def get_timefeatures(df):
    kurt=np.apply_along_axis(kurtosis, 1, pd.DataFrame(df))
    avg=np.apply_along_axis(np.mean, 1, pd.DataFrame(df))
    sd=np.apply_along_axis(np.std, 1, pd.DataFrame(df))
    skewness=np.apply_along_axis(scipy.stats.skew, 1, pd.DataFrame(df))
    sqr=np.apply_along_axis(np.square,1,df)
    eng=np.apply_along_axis(np.sum,1,sqr)
    autocorr=get_autocorr(df)
    #mean crossing rate
    #offset
    #iqr
    #min
    minimum=np.apply_along_axis(np.min,1,df)
    #max
    maximum=np.apply_along_axis(np.max,1,df)
    #Inter-Quartile Range
    interqr=get_iqr(df)
   
    #Collating all
    feature_set=pd.concat([pd.Series(kurt),pd.Series(avg),pd.Series(sd),pd.Series(skewness),pd.Series(eng),pd.Series(autocorr),pd.Series(minimum),pd.Series(maximum),pd.Series(interqr)],axis=1)
    return feature_set;


###################FREQ Domain#####################
    
def get_maxfreqfft(df):
    maxfreq_feature=[]
    minfreq_feature=[]
    sdfreq_feature=[]
    avgfreq_feature=[]
    df=pd.DataFrame(df)
    for col_id in range(0,df.shape[0]):
       curr_col=pd.Series(df.loc[col_id,:])
       maxfreq_feature.append(np.max(np.fft.fftfreq(len(curr_col))))
       minfreq_feature.append(np.min(np.fft.fftfreq(len(curr_col))))
       sdfreq_feature.append(np.std(np.fft.fftfreq(len(curr_col))))
       avgfreq_feature.append(np.mean(np.fft.fftfreq(len(curr_col))))
    feature_set=pd.concat([pd.Series(maxfreq_feature),pd.Series(minfreq_feature),pd.Series(avgfreq_feature),pd.Series(sdfreq_feature)],axis=1)
    return feature_set;  
    
    
    
def get_freqfeatures(df):
    fft_all=np.apply_along_axis(scipy.fftpack.rfft,1, pd.DataFrame(df))
    fft_dc=np.real(fft_all[:,0])
    #Getting real components
    freq_max=get_maxfreqfft(df)
    feature_set=pd.concat([pd.Series(fft_dc),pd.DataFrame(freq_max)],axis=1)
    return feature_set;

def choose_ml(classifier_name,train_x,train_y,test_x,test_y):
    if classifier_name=="lr":
        print("logistic regression")
        model=LogisticRegression(solver = 'liblinear',max_iter=1000)
        
    if classifier_name=="svm":
        print("support vector machine")
        params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        model=GridSearchCV(SVC(), params_grid, iid=True,cv=5)
   
    if classifier_name=="dt":
        print("decision tree")
        model=tree.DecisionTreeClassifier()
    
    if classifier_name=="rf":
        print("random forest")
        model=RandomForestClassifier(n_estimators=20,max_depth=10, random_state=42)
    
    if classifier_name=="ann":
        print("artificianl neural network")
        model=MLPClassifier(activation='logistic',hidden_layer_sizes=(train_x.shape[1],train_x.shape[1]+1,2),max_iter=500)
    
    if classifier_name=="nb":
        print("naive bayes")
        model=GaussianNB()
    
    if classifier_name=="knn":
        print("k  nearest neighbours")
        model=KNeighborsClassifier()
    
    if classifier_name=="xgb":
        print("xgboost")
        model=xgb.XGBRFClassifier()

    
    
    model.fit(train_x,train_y)
    pred_y=model.predict(test_x)
    accuracy=metrics.accuracy_score(test_y,pred_y)
    f1score=metrics.f1_score(test_y,pred_y)
    print("we are here")
    return accuracy,f1score;
    



data_gsr=pd.read_csv(r"C:\Users\Shikhav\Documents\Ubiattention_data\gsr_train.csv",header=None)
data_rr=pd.read_csv(r"C:\Users\Shikhav\Documents\Ubiattention_data\rr_train.csv",header=None)
data_hr=pd.read_csv(r"C:\Users\Shikhav\Documents\Ubiattention_data\hr_train.csv",header=None)
data_temp=pd.read_csv(r"C:\Users\Shikhav\Documents\Ubiattention_data\temp_train.csv",header=None)
labels_train=pd.read_csv(r"C:\Users\Shikhav\Documents\Ubiattention_data\labels_train.csv",header=None).iloc[:,0]

#min max scaling
#scaler= MinMaxScaler()
#data_gsr=scaler.fit_transform(data_gsr)
#data_rr = scaler.fit_transform(data_rr)
#data_hr = scaler.fit_transform(data_hr)
#data_temp = scaler.fit_transform(data_temp)

#GSR
gsr_time_features=get_timefeatures(data_gsr)
gsr_freq_features=get_freqfeatures(data_gsr)

gsr_all_features=pd.concat([gsr_time_features,gsr_freq_features],axis=1)
#RR
rr_time_features=get_timefeatures(data_rr)
rr_freq_features=get_freqfeatures(data_rr)
rr_all_features=pd.concat([rr_time_features,rr_freq_features],axis=1)

#HR
hr_time_features=get_timefeatures(data_hr)
hr_freq_features=get_freqfeatures(data_hr)
hr_all_features=pd.concat([hr_time_features,hr_freq_features],axis=1)

#Temp
temp_time_features=get_timefeatures(data_temp)
temp_freqfeatures=get_freqfeatures(data_temp)
temp_all_features=pd.concat([temp_time_features,temp_freqfeatures],axis=1)


ghrt_all_features=pd.concat([gsr_all_features,hr_all_features,rr_all_features,temp_all_features],axis=1,ignore_index=True)
ghrt_xy=pd.concat([ghrt_all_features,labels_train],axis=1)
ghrt_xy=ghrt_xy.dropna()
col_seq=list(range(0,ghrt_xy.shape[1]))
ghrt_xy.columns=col_seq
ghrt_x=ghrt_xy.iloc[:,range(0,ghrt_xy.shape[1]-1)]
ghrt_y=ghrt_xy.iloc[:,ghrt_xy.shape[1]-1]
#Train test split
ghrt_train_x,ghrt_test_x,train_y,test_y = train_test_split(ghrt_x,ghrt_y, test_size=0.20, random_state=42)


#model=xgb.XGBRFClassifier()

#model=tree.DecisionTreeClassifier()
#model=LogisticRegression(solver = 'liblinear',max_iter=1000)        
#model.fit()
#model.fit(ghrt_train_x,train_y)
#pred_y=model.predict(ghrt_test_x)     

user_classifier=input("enter classifier name:")
acc,f1=choose_ml(user_classifier,ghrt_train_x,train_y,ghrt_test_x,test_y)

print("acc is",acc)
print("f1 is",f1)
