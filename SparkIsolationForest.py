# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 10:03:35 2018

@author: yuexian
"""


import numpy as np
import pandas as pd
import random
import math
import json
import pickle
import argparse
import pyspark

from Node import Node

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext
from pyspark.sql.types import IntegerType, FloatType

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--filePath', type=str, default="data.csv")
parser.add_argument('-f', '--fileFeatures', type=str, default="features.json")
parser.add_argument('-t', '--nb_trees', type=int, default=100)
parser.add_argument('-s', '--nb_samples', type=int, default=256)

args = parser.parse_args()

sc = SparkContext()
sqlContext = SQLContext(sc)

def subsamplingRDD(X_train, sample_ratio):
    return X_train.sample(withReplacement=False,fraction=sample_ratio, seed=None)


def itree(X, e=0, limit=11):
    '''
    X: data set to construct the tree
    e: current tree height
    l: limit of the height of the tree
    '''
    m, n = X.shape    
    if e >= limit or m <= 1:    
        return Node([], [], None, None, m)
    
    rnd_attr = random.randint(0, n-1)  # choose one attribute by idx
    maxvalue = np.max(X[:, rnd_attr])
    minvalue = np.min(X[:, rnd_attr])
    rnd_split = random.random()*(maxvalue - minvalue + 1e-10) + minvalue  # get a value [minvalue, maxvalue)
    
    X_left = X[X[:, rnd_attr] < rnd_split]
    X_right = X[X[:, rnd_attr] >= rnd_split]
    e = e+1
        
    return Node(itree(X_left, e, limit), itree(X_right, e, limit), rnd_attr, rnd_split)
    

def itreeRDD(X, e=0, limit=10):
    '''
    X: RDD sampled data set to construct the tree, header not included
    e: current tree height
    l: limit of the height of the tree
    '''
    # m : the number of samples in X, n: number of columns
    X = sc.parallelize(X)

    m = X.count()
    
    if e > limit or m <= 1:    
        return Node([], [], None, None, m)
    
    n = X.take(1)[0].size
    rnd_attr = random.randint(0, n-1)  # choose one attribute by idx
    column = X.map(lambda line:line[rnd_attr])
    maxvalue = column.max()
    minvalue = column.min()
    rnd_split = random.random()*(maxvalue - minvalue + 1e-10) + minvalue  # get a value [minvalue, maxvalue)
    
    X_left = X.filter(lambda line:line[rnd_attr]<rnd_split)
    X_right = X.filter(lambda line:line[rnd_attr]>=rnd_split)

    #path_left = path.copy() + [0]
    #path_right = path.copy() + [1]
    
    #print("at height", e, "X --> Xleft.length =", X_left.shape[0], "Xright.length =", X_right.shape[0])
    return Node(itreeRDD(X_left, e+1, limit), itreeRDD(X_right, e+1, limit), rnd_attr, rnd_split, -1)
    

# create a forest RDD: seems pyspark doesn't support nested function， whether use itree or replace by forest.join(X_train)
# https://stackoverflow.com/questions/42325496/how-to-use-spark-intersection-by-key-or-filter-with-two-rdd

def iforestRDD(X_train, num_trees=150, subsample_size=265):
    """
    X_train: RDD data set to construct the forest
    """
    m = X_train.count()
    #n = X_train.take(1)[0].size

    # this list is perhaps too large and will cause a warning and maybe stop spark:
    # WARN scheduler.TaskSetManager: Stage 103 contains a task of very large size (564 KB). The maximum recommended task size is 100 KB.
    lis = []
    for i in range(num_trees):
        lis.append(np.array(subsamplingRDD(X_train, float(subsample_size)/m).collect()))
    # set numSlices to be large enough, the larger the numSlices, the smaller transfered to each executor
    # You can specify larger amounts through the commandline, for example with --executor-memory 64G
    forest = sc.parallelize(lis, numSlices=100)
    
    # we can also allow the limit to be greater when the sample size is larger
    height_limit = np.ceil(np.log2(subsample_size))
    trainforest = forest.map(lambda s:itree(s, e=0, limit=height_limit))

    return trainforest

    
def cost(num_items):
    #二叉搜索树的平均路径长度。0.5772156649:欧拉常数
    return int(2*(np.log(num_items-1) + 0.5772156649)-(2*(num_items-1)/num_items))


def get_path_RDD(x, T, e=0): 
    """
    x : a list or np.array, the row of RDD
    T: a Node instance
    e: recursive variable to record the length of a path
    """
    if T.size == 1 or T.size == 0:  # if tree node is a leaf
        return e
    elif T.size > 1:
        return e + cost(T.size)
    else: # if tree has children
        e += 1
        return get_path_RDD(x, T.left, e) if x[T.splitattr] < T.splitvalue else get_path_RDD(x, T.right, e)
    
# predict one line, forest here has been broadcasted
def predict_path(x, forest):
    predictions = np.array(list(map(lambda T:get_path_RDD(x, T), forest.value)))  # path length rdd
    return predictions

def paths_to_scores(predictions, num_samples):
    return math.pow(2, -float(predictions.sum()/predictions.size)/cost(num_samples)) #Anomaly Score rdd

def predict_score(x, forest, num_samples):
    predictions = np.array(list(map(lambda T:get_path_RDD(x, T), forest.value)))
    return math.pow(2, -float(predictions.sum()/predictions.size)/cost(num_samples)) #Anomaly Score rdd

###############################################################################

JSON_FILE = 'metadata.json'
with open(JSON_FILE, 'r') as f:
    metadata = json.load(f)
INDEX = metadata["index_name"]

def main():
    #hiveCtx = HiveContext(sc)

    # get the data
    filename = args.filePath
    nb_trees = args.nb_trees
    num_samples = args.nb_samples
    feature_path = args.fileFeatures

    print("get the features, if features.json has already been got by hdfs command") 
    with open(feature_path, 'r') as handle:
        features = json.load(handle)

    print("read the data from hdfs, by sc.textFile()")
    dist_data = sc.parallelize([])   
    dist_data = sc.textFile(filename, use_unicode="utf-8", minPartitions=1000)  # also possible to use rdd.repartition(128)

    # get the header, first action
    header = dist_data.first().split(",")
    # remove the header from dist data
    data = dist_data.filter(lambda line:line[:5] != "merge").map(lambda line:line.split(","))
    # create a data frame spark, not used very often, will stored in te disk only
    df_data = sqlContext.createDataFrame(data, schema=header).persist(pyspark.StorageLevel.DISK_ONLY)
    # get the selected columns
    isof_data = df_data.select(features["isof_features"])  # selected_columns_categorical+selected_columns_numerical
    # get the rdd for the selected columns, we transform each row to an np array; used very often put it to MEMORY_ONLY or MEMORY_AND_DISK
    data = isof_data.rdd.map(lambda line:np.array(list(map(lambda x:float(x), line)))).persist(pyspark.StorageLevel.MEMORY_AND_DISK)
    """
    # another way to transform data type from string to float 
    data_df = isof_data
    for col in data_df.columns:
        data_df=data_df.withColumn(col, data_df[col].cast(FloatType()))
    """

    print("construct the forest: a list of Node object defined in the Node.py, do the action collect() now ...")
    forest = iforestRDD(data, nb_trees, num_samples).collect()
    print("forest is calculated, save it to forest.pkl ...")
    with open("forest.pkl", "wb") as handle:
        pickle.dump(forest, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("forest pickled and saved.")

    print()
    print()
    
    print("start prediction process of isolation forest, each sample will pass all the trees in the forest ...")
    print("predict the path length: return a matrix of length, each column for each tree")
    print("broad cast the variable forest (large size), to make it cached on the machine rather than ship with tasks")
    forest = sc.broadcast(forest)
    path_mat_rdd = data.map(lambda line: predict_path(line, forest))
    
    print("the path length matrix is transformed to numpy array")
    path_mat = np.array(path_mat_rdd.collect())
    print("get average path length and the scores bathed on the path mat")
    path_mean = path_mat.mean(axis=1)

    print("predict the score of anomaly for each sample ")
    scores = np.array(list(map(lambda line: paths_to_scores(line, num_samples), path_mat)))
    #scores_rdd = data.map(lambda line: predict_score(line, forest, num_samples))
    # scores_mat is a rdd, calculate it and store it to csv
    #scores = np.array(scores_rdd.collect())
    
    print("store it to numpy data frame and write to csv ...")
    index_rdd = df_data.select(INDEX).rdd.map(list)  # df_data has been persisted
    indexes = pd.Series(np.array(index_rdd.collect()).ravel(), name=INDEX)   # add name to the index column,creating a series
    
    scoring_table_df = pd.DataFrame(index=indexes)  # Create scoring_table
    scoring_table_df['isof_path_depth'], scoring_table_df['score_isof'] = path_mean, scores  # Score of isof

    print("write result to isof.csv ...")
    scoring_table_df.to_csv("isof.csv", sep=';', encoding='utf-8', index=True)
    
    print("csv closed.")
    
if __name__ == "__main__":
    main()
    
    
    
    