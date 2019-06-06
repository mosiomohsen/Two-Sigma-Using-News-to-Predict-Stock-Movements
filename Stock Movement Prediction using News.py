import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# load data from twosigma
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(marketdf, newsdf) = env.get_training_data()

# import needed libraries 
import lightgbm as lgb
import matplotlib.pyplot as plt
from datetime import datetime, date
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import time

# data shape
print(marketdf.shape, newsdf.shape)

# preprocessing: replace outliers, which are the one with radical changes over one day, with mean values
marketdf['close_to_open'] =  np.abs(marketdf['close'] / marketdf['open'])
marketdf['assetName_mean_open'] = marketdf.groupby('assetName')['open'].transform('mean')
marketdf['assetName_mean_close'] = marketdf.groupby('assetName')['close'].transform('mean')
# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.
for i, row in marketdf.loc[marketdf['close_to_open'] >= 2].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        marketdf.iloc[i,5] = row['assetName_mean_open']
    else:
        marketdf.iloc[i,4] = row['assetName_mean_close']

for i, row in marketdf.loc[marketdf['close_to_open'] <= 0.5].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        marketdf.iloc[i,5] = row['assetName_mean_open']
    else:
        marketdf.iloc[i,4] = row['assetName_mean_close']
marketdf.drop(['close_to_open','assetName_mean_open','assetName_mean_close'], axis=1, inplace=True)
print("market data frame size: ", marketdf.shape)

# preprocessing: filter out strange data: platoe data or unknown assetcode
marketdf = marketdf[~marketdf.assetCode.isin(marketdf[marketdf.assetName=='Unknown'].assetCode.unique())]
marketdf = marketdf[marketdf.assetCode != 'PGN.N']
marketdf = marketdf[marketdf.assetCode != 'TW.N']
marketdf = marketdf[marketdf.assetCode != 'QRVO.O']
marketdf = marketdf[marketdf.assetCode != 'TECD.O']
print("market data frame size: ", marketdf.shape)

def prepare_data(marketdf, newsdf):
    # feature engineering
    # filter pre-2012 data because of unnormal behavior
    marketdf['time'] = marketdf.time.dt.strftime("%Y%m%d").astype(int)
    newsdf['time'] = newsdf.time.dt.strftime("%Y%m%d").astype(int)
    marketdf = marketdf.loc[marketdf['time'] > 20120000]
    newsdf = newsdf.loc[newsdf['time'] > 20120000]
    marketdf.sort_values(by=['time'])
    # market data 
    marketdf['average'] = (marketdf['close'] + marketdf['open'])/2
    marketdf['day_return'] = marketdf['close'] / marketdf['open']
    marketdf['prev_day_return'] = marketdf['returnsClosePrevMktres1'] / marketdf['returnsOpenPrevMktres1']
    marketdf['prev_10day_return'] = marketdf['returnsClosePrevMktres10'] / marketdf['returnsOpenPrevMktres10']
    marketdf['price_volume'] = marketdf['volume'] * marketdf['close']
    marketdf['volume_to_mean'] = marketdf['volume'] / marketdf['volume'].mean()
    marketdf['total_MACD'] = marketdf['close'].rolling(window=12).mean() - marketdf['close'].rolling(window=26).mean()
    marketdf['total_zscore'] = (marketdf['close']-marketdf['close'].rolling(window=200, min_periods=20).mean())/marketdf['close'].rolling(window=200, min_periods=20).std()
    
    # news data
    newsdf['assetCode'] = newsdf['assetCodes'].map(lambda x: list(eval(x))[0])
    newsdf['position'] = newsdf['firstMentionSentence'] / newsdf['sentenceCount']
    newsdf['sentence_word_count'] =  newsdf['sentenceCount'] / newsdf['wordCount']
    # apply tf-idf on  news title
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from nltk.corpus import stopwords
    #the top hundred words.
    vectorizer = CountVectorizer(max_features=500, stop_words={"english"})
    #we do this with TF-IDF.
    X = vectorizer.fit_transform(newsdf['headline'].values)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X)
    X_train_tf = tf_transformer.transform(X)
    X_train_vals = X_train_tf.mean(axis=1)
    del vectorizer
    del X
    del X_train_tf
    #mean tf-idf score for news article.
    d = pd.DataFrame(data=X_train_vals)
    newsdf['tf_score'] = d
    # drop extra junk from data
    droplist = ['sourceTimestamp','firstCreated','sourceId','headline','takeSequence','provider','firstMentionSentence',
                'sentenceCount','bodySize','headlineTag','marketCommentary','subjects','audiences','sentimentClass',
                'assetName', 'assetCodes','urgency','wordCount','sentimentWordCount']
    newsdf.drop(droplist, axis=1, inplace=True)
    marketdf.drop(['assetName', 'volume'], axis=1, inplace=True)
    # combine multiple news reports for same assets on same day
    newsgp = newsdf.groupby(['time','assetCode'], sort=False).aggregate(np.mean).reset_index()
    # join news reports to market data, note many assets will have many days without news data
    return pd.merge(marketdf, newsgp, how='left', on=['time', 'assetCode'], copy=False) #, right_on=['time', 'assetCodes'])

print('preparing data...')
cdf = prepare_data(marketdf, newsdf)    
# add binary target variable
cdf['binary_returnsOpenNextMktres10'] = (cdf['returnsOpenNextMktres10'] > 0).astype(int)
del marketdf, newsdf  # save the memory
print(cdf.shape)

# additional feature engineering: remove outlier, scaling
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures, MinMaxScaler, RobustScaler
def remove_outlier(df):
    low = .01
    high = .99
    quant_df = df.quantile([low, high])
    for name in list(df.columns):
        if is_numeric_dtype(df[name]) and name in ["close", "open", "average"]:
            df = df[(df[name] > quant_df.loc[low, name]) & (df[name] < quant_df.loc[high, name])]
    return df


# train-val split
def get_input(cdf, option):
    # time series slice
    dates = cdf['time'].unique()
    train_dates = range(len(dates))[:int(0.85*len(dates))]
    val_dates = range(len(dates))[int(0.85*len(dates)):]
    # train cols
    traincols = [col for col in cdf.columns if col not in ['time', 'assetCode', 'universe','binary_returnsOpenNextMktres10', 'returnsOpenNextMktres10']]
    # build, x, y, t, u, d
    if option == "train":
        X = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[train_dates])].values
        Y = cdf['binary_returnsOpenNextMktres10'].fillna(0).loc[cdf['time'].isin(dates[train_dates])].values
        r = cdf['returnsOpenNextMktres10'].fillna(0).loc[cdf['time'].isin(dates[train_dates])].values
        u = cdf['universe'].loc[cdf['time'].isin(dates[train_dates])]
        d = cdf['time'].loc[cdf['time'].isin(dates[train_dates])]
    elif option == "val":
        X = cdf[traincols].fillna(0).loc[cdf['time'].isin(dates[val_dates])].values
        Y = cdf['binary_returnsOpenNextMktres10'].fillna(0).loc[cdf['time'].isin(dates[val_dates])].values
        r = cdf['returnsOpenNextMktres10'].fillna(0).loc[cdf['time'].isin(dates[val_dates])].values
        u = cdf['universe'].loc[cdf['time'].isin(dates[val_dates])]
        d = cdf['time'].loc[cdf['time'].isin(dates[val_dates])]
    return X,Y,r,u,d
    
# r, u and d are used to calculate the scoring metric
print('building training and validation set...')
Xt,Yt,r_train,u_train,d_train = get_input(cdf, "train")
Xv,Yv,r_val,u_val,d_val = get_input(cdf, "val")
print(Xt.shape, Yt.shape)
print(Xv.shape, Yv.shape)

# Modeling: LightGBM
print ('Training lightgbm')

# parameters
params = {"objective" : "binary",
          "metric" : "binary_logloss",
          "num_leaves" : 600,
          "max_depth": -1,
          "learning_rate" : 0.001,
          "bagging_fraction" : 0.9,  # subsample
          "feature_fraction" : 0.9,  # colsample_bytree
          "bagging_freq" : 5,        # subsample_freq
          "bagging_seed" : 2018,
          "verbosity" : -1 }

lgtrain, lgval = lgb.Dataset(Xt, Yt), lgb.Dataset(Xv, Yv)
lgbmodel = lgb.train(params, lgtrain, 1500, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, verbose_eval=10)

# see the distribution of prediction
predicted_return = lgbmodel.predict(Xv, num_iteration=lgbmodel.best_iteration) * 2 - 1

# magic scaling function
def post_scaling(df):
    mean, std = np.mean(df), np.std(df)
    df = (df - mean)/ (std * 8)
    return np.clip(df,-1,1)

scaled_predicted_return = post_scaling(predicted_return)

# calculation of actual metric that is used to calculate final score
r_val = r_val.clip(-1,1) # get rid of outliers.
#x_t_i = predicted_return * r_val * u_val
x_t_i = scaled_predicted_return * r_val * u_val
data = {'day' : d_val, 'x_t_i' : x_t_i}
df = pd.DataFrame(data)
x_t = df.groupby('day').sum().values.flatten()
mean = np.mean(x_t)
print('mean', mean)
std = np.std(x_t)
print('std', std)
score_valid = mean / std
print('Validation score', score_valid)
# generate predictions on testing data
print("generating predictions...")
preddays = env.get_prediction_days()
traincols = [col for col in cdf.columns if col not in ['time', 'assetCode', 'universe','binary_returnsOpenNextMktres10', 'returnsOpenNextMktres10']]
i =0 
for marketdf, newsdf, predtemplatedf in preddays:
    print(i)
    i+=1
    cdf = prepare_data(marketdf, newsdf)
    Xp = cdf[traincols].fillna(0).values
    preds = lgbmodel.predict(Xp, num_iteration=lgbmodel.best_iteration) * 2 - 1
    predsdf = pd.DataFrame({'ast':cdf['assetCode'],'conf':post_scaling(preds)})
    predtemplatedf['confidenceValue'][predtemplatedf['assetCode'].isin(predsdf.ast)] = predsdf['conf'].values
    env.predict(predtemplatedf)
    
env.write_submission_file()
