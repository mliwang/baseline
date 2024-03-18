# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:21:10 2022

@author: 86138
"""
import math
data_name="ml25"
if data_name=="ml25":
    DATA_PATHA = "../data/ml25"
    information = DATA_PATHA+"/information.pkl"
    squenceLen = 80
    userpool_size = 1000
    news_size = 200
    content_dim=128
    influence_dim = 2
    # newsPAD=15715
    # UserPAD=38972
   
    news_influence=DATA_PATHA+"/process/news_influence.txt"
    emb_dir=DATA_PATHA+"/process/bert.pkl"
    cascade_train = DATA_PATHA+"/process/cascade_train.txt"
    cascade_val = DATA_PATHA+"/process/cascade_val.txt"
    cascade_test = DATA_PATHA+"/process/cascade_test.txt"
    shortestpath_train = DATA_PATHA+"/process/shortestpath_train.txt"
    shortestpath_val = DATA_PATHA+"/process/shortestpath_val.txt"
    shortestpath_test = DATA_PATHA+"/process/shortestpath_test.txt"
    val_pkl=DATA_PATHA+"/process/val.pkl"
    train_pkl=DATA_PATHA+"/process/train.pkl"
    test_pkl=DATA_PATHA+"/process/test.pkl"
    observation = 24*365*10*60*60-1
    pre_times = [1*24 * 3600]
    newsgraph=DATA_PATHA+"/process/news_news.pkl"
    news_dict=DATA_PATHA+"/process/news_dict.pkl"
    user_dict = DATA_PATHA + "/process/user_dict.pkl"
    n_time_interval = 1*60*60
    # print ("the number of time interval:",n_time_interval)
    time_interval = math.ceil((observation+1)*1.0/n_time_interval)#向上取整
elif data_name=="twitter":
    DATA_PATHA = "../data/twitter"
    information = DATA_PATHA+"/information.pkl"
    squenceLen = 15
    userpool_size = 100
    news_size = 10
    content_dim=128
    influence_dim=3
    newsPAD=-1
    UserPAD=-1
    cascades  = DATA_PATHA+"/raw/repost_data.txt"
    news_influence=DATA_PATHA+"/process/news_influence.txt"
    emb_dir=DATA_PATHA+"/process/bert.pkl"
    cascade_train = DATA_PATHA+"/process/cascade_train.txt"
    cascade_val = DATA_PATHA+"/process/cascade_val.txt"
    cascade_test = DATA_PATHA+"/process/cascade_test.txt"
    shortestpath_train = DATA_PATHA+"/process/shortestpath_train.txt"
    shortestpath_val = DATA_PATHA+"/process/shortestpath_val.txt"
    shortestpath_test = DATA_PATHA+"/process/shortestpath_test.txt"
    val_pkl=DATA_PATHA+"/process/val.pkl"
    train_pkl=DATA_PATHA+"/process/train.pkl"
    test_pkl=DATA_PATHA+"/process/test.pkl"
    observation = 1*60-1
    pre_times = [24 * 3600]
    newsgraph=DATA_PATHA+"/process/news_news.pkl"
    news_dict=DATA_PATHA+"/process/news_dict.pkl"
    user_dict = DATA_PATHA + "/process/user_dict.pkl"
    n_time_interval = 1*60*60
    # print ("the number of time interval:",n_time_interval)
    time_interval = math.ceil((observation+1)*1.0/n_time_interval)#向上取整
elif data_name=="aminer":
    DATA_PATHA = "data/aminer"
    information = DATA_PATHA+"/information.pkl"
    squenceLen = 80
    userpool_size = 1000
    news_size = 200
    content_dim=128
    influence_dim = 2
    newsPAD=15715
    UserPAD=38972
    cascades  = DATA_PATHA+"/raw/repost_data.txt"
    news_influence=DATA_PATHA+"/process/news_influence.txt"
    emb_dir=DATA_PATHA+"/process/bert.pkl"
    cascade_train = DATA_PATHA+"/process/cascade_train.txt"
    cascade_val = DATA_PATHA+"/process/cascade_val.txt"
    cascade_test = DATA_PATHA+"/process/cascade_test.txt"
    shortestpath_train = DATA_PATHA+"/process/shortestpath_train.txt"
    shortestpath_val = DATA_PATHA+"/process/shortestpath_val.txt"
    shortestpath_test = DATA_PATHA+"/process/shortestpath_test.txt"
    val_pkl=DATA_PATHA+"/process/val.pkl"
    train_pkl=DATA_PATHA+"/process/train.pkl"
    test_pkl=DATA_PATHA+"/process/test.pkl"
    observation = 24*1*60*60-1
    pre_times = [24 * 3600]
    newsgraph=DATA_PATHA+"/process/news_news.pkl"
    news_dict=DATA_PATHA+"/process/news_dict.pkl"
    user_dict = DATA_PATHA + "/process/user_dict.pkl"
    n_time_interval = 1*60*60
    # print ("the number of time interval:",n_time_interval)
    time_interval = math.ceil((observation+1)*1.0/n_time_interval)#向上取整