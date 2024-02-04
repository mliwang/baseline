import math
data_name="aminer"
if data_name=="weibo":
    DATA_PATHA = "../data"
    cascades  = DATA_PATHA+"/dataset_weibo.txt"
    
    cascade_train = DATA_PATHA+"/cascade_train.txt"
    cascade_val = DATA_PATHA+"/cascade_val.txt"
    cascade_test = DATA_PATHA+"/cascade_test.txt"
    shortestpath_train = DATA_PATHA+"/shortestpath_train.txt"
    shortestpath_val = DATA_PATHA+"/shortestpath_val.txt"
    shortestpath_test = DATA_PATHA+"/shortestpath_test.txt"
    
    observation = 3*60*60-1
    pre_times = [24 * 3600]
elif data_name=="aminer":
    DATA_PATHA = "../data/aminer"
    cascades  = DATA_PATHA+"/dataset_weibo.txt"
    
    cascade_train = DATA_PATHA+"/process/cascade_train.txt"
    cascade_val = DATA_PATHA+"/process/cascade_val.txt"
    cascade_test = DATA_PATHA+"/process/cascade_test.txt"
    shortestpath_train = DATA_PATHA+"/process/shortestpath_train.txt"
    shortestpath_val = DATA_PATHA+"/process/shortestpath_val.txt"
    shortestpath_test = DATA_PATHA+"/process/shortestpath_test.txt"
    train_pkl = "data/aminer/data_train.pkl"
    val_pkl = "data/aminer/data_val.pkl"
    test_pkl = "data/aminer/data_test.pkl"
    information = "data/aminer/information.pkl"
    observation = 3*60*60-1
    print ("observation time",observation)
    n_time_interval = 6
    print ("the number of time interval:",n_time_interval)
    time_interval = math.ceil((observation+1)*1.0/n_time_interval)#向上取整
    print ("time interval:",time_interval)
    lmax = 2
elif data_name=="twitter":
    DATA_PATHA = "data/twitter"
    information = DATA_PATHA+"/information.pkl"
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
    n_time_interval = 1*60*60
    # print ("the number of time interval:",n_time_interval)
    time_interval = math.ceil((observation+1)*1.0/n_time_interval)#向上取整
    print("time_interval:", time_interval)