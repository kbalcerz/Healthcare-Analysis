
# coding: utf-8

# In[2]:


from google.cloud import storage
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans 
import pandas as pd
import numpy as np


# In[3]:


def lr_prediction(cms_df):
    
    cms_df = cms_df.fillna(0)
    X = cms_df.drop(['provider_id','facility_name','street_address', 'city', 'state','drg_definition','hospital_referral_region_description','apc','hospital_referral_region',
           'zip_code','total_snf_charge_amount','total_snf_medicare_allowed_amount','total_snf_medicare_payment_amount','total_snf_medicare_standard_payment_amount'],axis=1)
    # X = cms_df[['']]
    Y = cms_df[['total_snf_medicare_standard_payment_amount']]

    import math
    X_train = X[:math.ceil((80*len(X))/100)]
    Y_train = Y[:math.ceil((80*len(Y))/100)]

    X_test = X[math.ceil((80*len(X))/100):]
    Y_test = Y[math.ceil((80*len(Y))/100):]
    
    lr_model = LinearRegression(normalize=True, n_jobs = -1)
    lr_x_train = np.array(X_train[['total_stays','average_estimated_submitted_charges','outpatient_services']])
    lr_y_train = np.array(Y_train)
    lr_model.fit(lr_x_train,lr_y_train)
    
    y_pred = lr_model.predict(np.array(X_test[['total_stays','average_estimated_submitted_charges','outpatient_services','average_total_payments']]))
    
    Y_test['LR_predictions'] = y_pred
    Y_test.LR_predictions = Y_test.LR_predictions.astype('int64')
    Y_test[:100].plot(kind='line')

# In[4]:


def rf_prediction(cms_df):
    cms_df = cms_df.fillna(0)
    X = cms_df.drop(['provider_id','facility_name','street_address', 'city', 'state','drg_definition','hospital_referral_region_description','apc','hospital_referral_region',
           'zip_code','total_snf_charge_amount','total_snf_medicare_allowed_amount','total_snf_medicare_payment_amount','total_snf_medicare_standard_payment_amount'],axis=1)

    Y = cms_df[['total_snf_medicare_standard_payment_amount']]

    import math
    X_train = X[:math.ceil((80*len(X))/100)]
    Y_train = Y[:math.ceil((80*len(Y))/100)]

    X_test = X[math.ceil((80*len(X))/100):]
    Y_test = Y[math.ceil((80*len(Y))/100):]


    
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 0) 
    rf.fit(X_train, Y_train) 
    predicted_labels = rf.predict(X_test)

    import pandas as pd
    feature_importances = pd.DataFrame(rf.feature_importances_,
                                       index = X_train.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)
    
    feature_importances.plot.bar()
    
    Y_test['predictions'] = predicted_labels
    Y_test.predictions = Y_test.predictions.astype('int64')
    
    ax = Y_test[:100].plot(kind='line')
    ax.set_xlabel('Providers')
    ax.set_ylabel('Cost for 2014 (USD)')
    ax.set_title('Actual vs predicted cost per hospital')


# In[5]:


def kmeans_clustering(cms_df):
    cms_df = cms_df.fillna(0)
    kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=500,n_init=50,random_state=2) 
    X = cms_df.drop(['provider_id','facility_name','street_address', 'city', 'state','drg_definition','hospital_referral_region_description','apc','hospital_referral_region',
       'zip_code','total_snf_medicare_allowed_amount','total_snf_medicare_payment_amount'],axis=1)
    X['clusters'] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    #Plot the clusters obtained using k means
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(X['distinct_beneficiaries_per_provider'],X['outpatient_services'],
                         c=X['clusters'],s=50)
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('male_beneficiaries')
    ax.set_ylabel('total_snf_charge_amount')
    plt.colorbar(scatter)
    
    mean_hcc = pd.DataFrame()
    print(X['average_hcc_score'].loc[X['clusters'] == 1].mean())
    mean_hcc['cluster_1'] = [X['average_hcc_score'].loc[X['clusters'] == 1].mean()]
    mean_hcc['cluster_2'] = [X['average_hcc_score'].loc[X['clusters'] == 2].mean()]
    mean_hcc['cluster_3'] = [X['average_hcc_score'].loc[X['clusters'] == 3].mean()]
    print(mean_hcc)
    ax_hcc = mean_hcc.plot(kind='bar')
    ax_hcc.set_xlabel('Clusters')
    ax_hcc.set_ylabel('average hcc score')
    
    mean_total_stays = pd.DataFrame()
    mean_total_stays['cluster_1'] = [X['total_stays'].loc[X['clusters'] == 1].mean()]
    mean_total_stays['cluster_2'] = [X['total_stays'].loc[X['clusters'] == 2].mean()]
    mean_total_stays['cluster_3'] = [X['total_stays'].loc[X['clusters'] == 3].mean()]
    
    ax_ts = mean_total_stays.plot(kind='bar')
    ax_ts.set_xlabel('Clusters')
    ax_ts.set_ylabel('total stays')
    
    mean_beneficiary = pd.DataFrame()
    mean_beneficiary['cluster_1'] = [X['distinct_beneficiaries_per_provider'].loc[X['clusters'] == 1].mean()]
    mean_beneficiary['cluster_2'] = [X['distinct_beneficiaries_per_provider'].loc[X['clusters'] == 2].mean()]
    mean_beneficiary['cluster_3'] = [X['distinct_beneficiaries_per_provider'].loc[X['clusters'] == 3].mean()]
    
    ax_b = mean_beneficiary.plot(kind='bar')
    ax_b.set_xlabel('Clusters')
    ax_b.set_ylabel('Distinct Beneficiaries Per Provider')
    
    mean_payment = pd.DataFrame()
    mean_payment['cluster_1'] = [X['average_age'].loc[X['clusters'] == 1].mean()]
    mean_payment['cluster_2'] = [X['average_age'].loc[X['clusters'] == 2].mean()]
    mean_payment['cluster_3'] = [X['average_age'].loc[X['clusters'] == 3].mean()]
    
    ax_p = mean_payment.plot(kind='bar')
    ax_p.set_xlabel('Clusters')
    ax_p.set_ylabel('Average age')
    
