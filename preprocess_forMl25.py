# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 10:58:27 2024

@author: wangranr
"""

# -*- coding: utf-8 -*-
"""
Created on 2023/3/22

@author: mliwang
"""

import time
import logging
import os
import config
import scipy.sparse
import gc
import networkx as nx
logging.basicConfig(filename='logger.log', level=logging.INFO)
import time
import numpy as np
import pandas as pd

import torch
import pickle
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from tqdm import tqdm
def gen_cascade_graph(observation_time,  # 这个时间控制观测时间的,1h这里是
                      pre_times,
                      filename,
                      filename_ctrain,
                      filename_cval,
                      filename_ctest,
                      filename_strain,
                      filename_sval,
                      filename_stest):
    """
    """

    file_ctrain = open(filename_ctrain, "w")
    file_cval = open(filename_cval, "w")
    file_ctest = open(filename_ctest, "w")
    file_strain = open(filename_strain, "w")
    file_sval = open(filename_sval, "w")
    file_stest = open(filename_stest, "w")

    repost = {}  # key为nid, value:(a,b)  a表示number_of_reposts b为一个list
    cp = 0  # 当前的微博id
    maxnr = 0  # max number_of_reposts
    Observation_time = []  # '2009-08-28 10:51:11'~ '2012-12-26 05:29:35'
    cascades_total = {}
    ispost = True
    rating=pd.read_csv(filename)
    
    # repost=dict(rating.groupby(rating['movieId'])[['userId','timestamp']])  
    # cascades_total=dict(rating.groupby(rating['movieId']).count())
    for index, row in rating.iterrows():
        cp=int(row['movieId'])
        if cp in repost:
            repost[cp].append((str(int(row['userId'])),int(row['timestamp'])))
            cascades_total[cp] =cascades_total[cp] +1
        else:
            repost[cp]=[(str(int(row['userId'])),int(row['timestamp']))]
            cascades_total[cp] =0

    # map: id -> type   1: train,  2: val,  3: test
    sorted_message_time = sorted(cascades_total.items(), key=None)
    cascades_type_train = {k: 1 for k, v in sorted_message_time[: int(len(cascades_total) * 14.0 / 20.0) + 1]}
    cascades_type_val = {k: 2 for k, v in sorted_message_time[int(len(cascades_total) * 14.0 / 20.0) + 1: int(
        len(cascades_total) * 17.0 / 20.0) + 1]}
    cascades_type_test = {k: 3 for k, v in sorted_message_time[int(len(cascades_total) * 17.0 / 20.0) + 1:]}
    cascades_type = {**cascades_type_train, **cascades_type_val, **cascades_type_test}
    pickle.dump(cascades_total, open(config.news_influence, 'wb'))


    #统计各个user的出现次数，将出现3次以下的user去掉
    ids = repost.keys()
    alluser={}
    for k,path in repost.items():
        for v in path:
            if v[0] in alluser.keys():
                alluser[v[0]]+=1
            else:
                alluser[v[0]]= 1
    print("原始用户数量;", len(alluser))
    alluser = dict(filter(lambda item: item[1] > 3, alluser.items())).keys()#保留的user
    print("保留的用户数量;",len(alluser))


    #     labels = [[0] * len(pre_times) for i in range(len(ids))]
    #     paths = [v for k,v in repost.items()]
    #     observation_paths = [[p[0] for p in path if p[1] < observation_time] for path in repost.values()]
    observation_paths = []  # 各个级联大图下的扩散图，每个子path为当前时间窗口新加的
    pols = []  # 装流行度随时间的变化
    labels = []
    nids = []
    # 获取各个时间段的流行度
    for nid in tqdm(ids):
        nidpaths = []
        paths = []
        popularity = [1]
        v = repost[nid]
        # print(v)
        # if len(v)<5:
        #     continue
        restart = True
        diff =[v[i][1]-v[i - 1][1] for i in range(1, len(v))]
        #         print("time scope:",repost[nid][-1][1]-starttime)
        for i in range(len(v)):
            t = v[i][1]
            u = v[i][0]
            if u not in alluser:
                continue
            if restart:
                starttime = t
                paths = [u]
                restart = False
                continue
            if t - starttime < observation_time:
                paths.append(u)
                if i==(len(v)-1) or(t - starttime + diff[i] >= observation_time) or (i!=0 and diff[i] > observation_time):
                    nidpaths.append(paths)
                    restart = True
        pol = [len(paths) for paths in nidpaths]

        polularity = []
        sump = 0
        for i in range(1, len(nidpaths)):
            sump = sump + pol[i]
            polularity.append(str(sump))
        if len(polularity) > 2:
            nids.append(str(nid))
            # print(polularity)
            pols.append(polularity)
            labels.append(nidpaths[-1])  # 最后还是给一个
            nidpaths = [",".join(paths) for paths in nidpaths]
            observation_paths.append(nidpaths)
            print(nidpaths)

    print("总样本数：",len(nids))

    # 划分数据集，包括扩散图，流行度
    strain_lines = [nids[i] + "\t" + ",".join(pols[i]) + "\t" + "\t".join(observation_paths[i]) for i in
                    range(len(nids)) if nids[i] in cascades_type and cascades_type[nids[i]] == 1]
    strain_lines = [line + ("" if line[-1] == '\n' else "\n") for line in strain_lines]
    sval_lines = [nids[i] + "\t" + ",".join(pols[i]) + "\t" + "\t".join(observation_paths[i]) for i in range(len(nids))
                  if nids[i] in cascades_type and cascades_type[nids[i]] == 2]
    sval_lines = [line + ("" if line[-1] == '\n' else "\n") for line in sval_lines]
    stest_lines = [nids[i] + "\t" + ",".join(pols[i]) + "\t" + "\t".join(observation_paths[i]) for i in range(len(nids))
                   if nids[i] in cascades_type and cascades_type[nids[i]] == 3]
    stest_lines = [line + ("" if line[-1] == '\n' else "\n") for line in stest_lines]
    ctrain_lines = [
        nids[i] + "\t" + str(pols[i][-1]) + "\t" + str(len(observation_paths[i])) + "\t" + " ".join(labels[i]) + "\n"
        for i in range(len(nids)) if nids[i] in cascades_type and cascades_type[nids[i]] == 1]
    cval_lines = [
        nids[i] + "\t" + str(pols[i][-1]) + "\t" + str(len(observation_paths[i])) + "\t" +" ".join(labels[i]) + "\n"
        for i in range(len(nids)) if nids[i] in cascades_type and cascades_type[nids[i]] == 2]
    ctest_lines = [
        nids[i] + "\t" + str(pols[i][-1]) + "\t" + str(len(observation_paths[i])) +"\t" + " ".join(labels[i]) + "\n" for i in
        range(len(nids)) if nids[i] in cascades_type and cascades_type[nids[i]] == 3]
    file_ctrain.writelines(ctrain_lines)
    file_cval.writelines(cval_lines)
    file_ctest.writelines(ctest_lines)
    file_strain.writelines(strain_lines)
    file_sval.writelines(sval_lines)
    file_stest.writelines(stest_lines)
    file_ctrain.close()
    file_cval.close()
    file_ctest.close()
    file_strain.close()
    file_sval.close()
    file_stest.close()
def readgraph(flename):
    '''
    读级联图
    '''
    graphs = {}
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            graphs[walks[0]] = []  # walk[0] = cascadeID
            for i in range(2, len(walks)):
                s = walks[i] # node list
                t = i-1  # time
                graphs[walks[0]].append([[str(xx) for xx in s.split(",")], int(t)])
    return graphs
def readPopularity(flename,squenceLen):
    '''
    读各个级联对应的流行度变化情况
    '''
    # popdict={}
    sampleNum=0#样本数量
    ids=[]
    X=[]
    Y=[]
    with open(flename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')
            cascadeID=walks[0]
            popseq=walks[1].split(",")
            popseq=[int(p) for p in popseq]
            #对各个序列进行填充，对齐。规则：经量取后面一截
            xx=[]
            # popdict[cascadeID] =[]# np.array(x)
            start=0
            i=squenceLen
            step=50
            while start<len(popseq) and i+1<len(popseq):
                x=popseq[start:i]
                y=popseq[i]
                ids.append(cascadeID)
                X.append(np.array(x))
                Y.append(y)
                # popdict[cascadeID].append((np.array(x),y))
                start=start+step
                i=start+squenceLen
            # popseq=popseq[:-1]
            if start<len(popseq) and len(popseq)<=i:
                x=popseq[start:-1]
                if len(x)<squenceLen:
                    x = x+[x[-1]] * (squenceLen - len(x))
                y = popseq[-1]
                # popdict[cascadeID].append((np.array(x), y))
                ids.append(cascadeID)
                X.append(np.array(x))
                Y.append(y)
    return ids,X,Y
def get_maxsize(sizes):
    max_size = 0
    for cascadeID in sizes:
        max_size = max(max_size,sizes[cascadeID])
    gc.collect()
    return max_size
#read label and size from cascade file
def read_labelANDsize(filename):
    lastuser = {}
    sizes = {}
    # pop={}
    with open(filename, 'r') as f:
        for line in f:
            profile = line.split('\t')
            lastuser[profile[0]] = profile[-1].replace("\n","").split(' ')[-1]#最后一步谁感染了， profile[-1].split(' ')[-1]取最后一个时间点的用户，根据任务的不同这个地方可以是多个
            sizes[profile[0]] = int(profile[2])#几个时间步
            # pop[profile[0]] =  int(profile[1])
    return lastuser,sizes#,pop
def get_original_ids(graphs):
    original_ids = set()
    for graph in graphs.keys():
        for walk in graphs[graph]:
            for i in walk[0]:
                original_ids.add(i)
    print ("length of original isd:",len(original_ids))
    return original_ids
def get_nodes(graph):
    nodes = {}
    j = 0
    for walk in graph:
        for i in walk[0]:
            if i not in nodes.keys():
                nodes[i] = j
                j = j+1
    return nodes
def get_max_node_num(graphs):
    max_num = 0
    for key,graph in graphs.items():
        nodes = get_nodes(graph)
        max_num = max(max_num,len(nodes))
    return max_num


# trans the original ids to 1~n
def newsidmap(original_ids,dir=config.news_dict):
    '''
    ''
    id做映射
    '''
    if os.path.exists(dir):
        with open(dir, 'rb') as f:
            saveitem = pickle.load(f)
            return saveitem[0], saveitem[1]
    else:
        or_new = {}  # 原始的到新的
        new_or = {}  # 新的到原始的
        for i in range(1,len(original_ids)+1):#0号留着padding
            or_new[original_ids[i-1]] = i
            new_or[i] = original_ids[i-1]
        pickle.dump((or_new, new_or), open(dir, 'wb'))
        return or_new, new_or
def generateDataset(squenceLen=200,userpool_size=300,news_size=200):
    ### 拿到级联图 data set ###
    graphs_train = readgraph(config.shortestpath_train)
    graphs_val = readgraph(config.shortestpath_val)
    graphs_test = readgraph(config.shortestpath_test)
    print("参与训练的总消息数：", len(graphs_train.keys()))
    print("参与测试的总消息数：", len(graphs_test.keys()))
    print("参与验证的总消息数：", len(graphs_val.keys()))

    ### get labels    都是带id的###
    lastuser_train, sizes_train= read_labelANDsize(config.cascade_train)#label_pop对应的是整个news在下个时间点的流行度
    lastuser_val, sizes_val = read_labelANDsize(config.cascade_val)
    lastuser_test, sizes_test= read_labelANDsize(config.cascade_test)
    NUM_SEQUENCE = max(get_maxsize(sizes_train), get_maxsize(sizes_val), get_maxsize(sizes_test))
    print("各个级联步骤中，最长的步骤为",NUM_SEQUENCE)


    ####get popularity  ###
    # pop_train=readPopularity(config.shortestpath_train, squenceLen)
    # pop_val = readPopularity(config.shortestpath_val, squenceLen)
    # pop_test = readPopularity(config.shortestpath_test, squenceLen)
    ####get 初始影响力 ###
    # influenceDict = pickle.load(open(config.news_influence, 'rb'))
    #看用户数，方便后面做映射
    max_num = max(get_max_node_num(graphs_train), get_max_node_num(graphs_test), get_max_node_num(graphs_val))#拿到各个级联最大的相关用户数量
    # get the total original_ids and tranform the index from 0 ~n-1
    # original_ids = get_original_ids(graphs_train) \
    #     .union(get_original_ids(graphs_val)) \
    #     .union(get_original_ids(graphs_test))
    # original_ids.add(-1)
    # index is new index
    index,_ =newsidmap(list(get_original_ids(graphs_train) \
        .union(get_original_ids(graphs_val)) \
        .union(get_original_ids(graphs_test))), dir=config.user_dict)
    # index = IndexDict(original_ids)#用户id的统一映射
    # gobal graph
    # 全局图
    # 新闻的id得处理Todo
    PAD = 0#newsPAD
    max_num=0#userPAD
    or2new, _ = newsidmap(list(set(graphs_train.keys()).union(set(graphs_val.keys())).union(set(graphs_test.keys()))),config.news_dict)
    golobal_G = ({},{}) # 这个地方得改，({},{})
    print("Create golobal graph!")
    ##1.这个地方必须先建立全局的二分图，后面才好拿2hop user 2hop news
    golobal_G = jointGraph(golobal_G, graphs_train, or2new)
    golobal_G = jointGraph(golobal_G, graphs_val, or2new)
    golobal_G = jointGraph(golobal_G, graphs_test, or2new)

    # print("Create train dataset!")
    # writeXY(PAD,or2new, graphs_train,pickle.load(open(config.news_influence, 'rb')),sizes_train,config.shortestpath_train, squenceLen, lastuser_train,index,
    #                               config.train_pkl, golobal_G,max_num,userpool_size,news_size)
    print("create val")
    writeXY(PAD,or2new, graphs_val, pickle.load(open(config.news_influence, 'rb')),sizes_val, config.shortestpath_val, squenceLen, lastuser_val, index,
            config.val_pkl, golobal_G,max_num,userpool_size,news_size)
    print("create val an test")
    writeXY(PAD,or2new, graphs_test,pickle.load(open(config.news_influence, 'rb')),sizes_test, config.shortestpath_test, squenceLen, lastuser_test, index,
                                  config.test_pkl, golobal_G,max_num,userpool_size,news_size)
    # 文本的表示，大换血
    print("Create new Embedding!")

    if os.path.exists(config.emb_dir):
        pass
    else:
        with open(config.DATA_PATHA + '/doc2vec_128.pkl', 'rb') as f:
            em = pickle.load(f)
            newdict = {}
            cannotfind = 0
            for id in or2new.keys():
                if int(id) in em.keys():
                    newdict[or2new[id]] = np.array(em[int(id)]).reshape(1, config.content_dim)
                else:
                    cannotfind = cannotfind + 1
            print("Warning we have %d news cannot find text  of  all %d data!"%(cannotfind,len(or2new.keys())))
            newdict[PAD] = np.random.rand(1, config.content_dim)  # 最大的id设为填充值

            emVec = dict(sorted(newdict.items(), key=lambda x: x[0]))
            emVec = np.stack(list(emVec.values()), 0)
            pickle.dump(emVec, open(config.emb_dir, 'wb'))
    return PAD,max_num
def jointGraph(golobal_G,graphs,or2new):
    '''
    将graphs中的数据并入golobal_G   graph里面user的id是str 就用原来的，news的就用int的新的，这样就可以避免冲突了
    or2new  news的id映射
    '''
    for key, graph in graphs.items():
        # 把user加进来，把news id加进来
        nodes_items = [str(w) for walk in graph for w in walk[0]]
        # nodes_items =[index[str(it)] for it in nodes_items]
        # if 1916 in nodes_items:
        #     print("Bug Done!")
        golobal_G[0][key] =nodes_items
        for u in nodes_items:
            if u in golobal_G[1].keys():
                golobal_G[1][u].append(key)
            else:
                golobal_G[1][u]=[key]
    return golobal_G

def writeXY(PAD,or2new,graphs,influenceDict,sizes,pop_traindir,squenceLen, lastuser, index,
                                  savedir, golobal_G,userPAD,userpool_size=300,news_size=200):
    '''
    最后写入文件的内容包括：
    流行度序列（id列表，input_seq, 影响用户数/粉丝数，，2hop的用户，各个用户相关的news,标签-流行度，标签-下一个item）
    当前新闻与2hop新闻的相似度在dataset里算
    '''
    ids,input_seq,Label_pop=readPopularity(pop_traindir, squenceLen)

    # ids=[k for k,v in pop_train.items() for _ in v]

    # input_seq = []  # 流行度序列
    influce_num = []   #影响用户数/粉丝数
    # sim_mat = []#相似度矩阵  这个太大了还是放在dataset 里面逐批算
    user_pool = [] #与各个新闻相关的2-hop用户列表
    news_pool = []  #
    # Label_pop = []
    Label_nextUser = []
    num_2hopuser=[]#用于数据统计
    #简化存储
    id_data=[or2new[i] for i in ids]
    influce_num=[np.array([influenceDict[i],sizes[i]]) for i in ids]
    Label_pop = [np.log(y + 1.0) / np.log(2.0) for y in Label_pop]
    del influenceDict,sizes
    gc.collect()
    from tqdm import tqdm
    user2hopdict={}
    news_pooldict={}
    Label_nextUserdict={}

    for key in tqdm(graphs.keys()):#两个任务拿到user_pool  news_pool Label_nextUser
        users = set([str(u) for walk in graphs[key] for u in walk[0]])
        hop1news = set([i for u in users for i in golobal_G[1][u] if i != key])#当前看了当前新闻的用户还看了什么新闻
        user2hop=set([u for i in hop1news for u in golobal_G[0][i] if u not in users])#
        del users, hop1news
        gc.collect()
        num_2hopuser.append(len(user2hop))
        from random import sample,shuffle
        if len(user2hop)>userpool_size:
            user2hop = sample(list(user2hop), userpool_size)  # 最有可能被进一步传播消息的用户
        #保证 lastuser[key]在 user2hop中
        if lastuser[key] not in user2hop:
            if len(user2hop)>=userpool_size:
                user2hop = sample(list(user2hop), userpool_size-1)
            user2hop.append(lastuser[key])
        # 拿到这些用户关注新闻的情况
        user_news = []
        for u in user2hop:
            news = [or2new[node] for node in golobal_G[1][u] if node != key]
            if len(news) < news_size:
                news = news + [PAD] * (news_size - len(news))
            else:
                news = sample(news, news_size)
            user_news.append(np.array(news))
        news_pooldict[key]=np.array(user_news).reshape(userpool_size,news_size)
        #把二阶候选用户加入数据集
        #user2hop<=userpool_size
        user2hop = [index[u] for u in user2hop]
        shuffle(user2hop)
        lastuserindex = index[lastuser[key]]#
        lastuserindex = user2hop.index(lastuserindex)
        if len(user2hop) <= userpool_size:
            user2hop=user2hop+[userPAD]*(userpool_size-len(user2hop))

        user2hopdict[key]=np.array(user2hop)
        # Label_pop.append(np.log(label_pop_train[key]+1.0)/np.log(2.0) )
        # Label_nextUser.append(lastuserindex)  # index[0][lastuser[key]]
        Label_nextUserdict[key]=lastuserindex
        #还差点，要变数据类型
    user_pool = [user2hopdict[i] for i in ids]
    news_pool = [news_pooldict[i] for i in ids]
    Label_nextUser = [Label_nextUserdict[i] for i in ids]
    print("2hopuser 的数量分布",max(num_2hopuser),min(num_2hopuser),np.mean(num_2hopuser))
    print("数据集样本数：",len(id_data))
    pickle.dump((id_data, input_seq, influce_num, user_pool, news_pool, Label_pop, Label_nextUser), open(savedir, 'wb'))
    return 

if __name__ =="__main__":
    datadir=config.DATA_PATHA
    if not os.path.exists(datadir+"/process"):
        os.makedirs(datadir+"/process")
    #下面的三步只要顺序执行即可！
    print(datadir+"/process")

    #1.生成级联
    gen_cascade_graph(config.observation,  # 这个时间控制观测时间的,1h这里是
                      config.pre_times,
                      "../../data/ML25M/ml-25m/ratings.csv",
                      os.path.join(datadir, "process/cascade_train.txt"),
                      os.path.join(datadir, "process/cascade_val.txt"),
                      os.path.join(datadir, "process/cascade_test.txt"),
                      os.path.join(datadir, "process/shortestpath_train.txt"),
                      os.path.join(datadir, "process/shortestpath_val.txt"),
                      os.path.join(datadir, "process/shortestpath_test.txt"))
    # # 2.获取内容的embedding
  
    # 3.将数据处理成模型训练的样子
    PAD,max_num=generateDataset(config.squenceLen, config.userpool_size, config.news_size)
    pickle.dump((PAD, max_num), open(config.information, 'wb'))
    print("News Pad number: ",PAD," User PAD",max_num,"。这两个参数记得设置到opt.newsPAD里面！")

