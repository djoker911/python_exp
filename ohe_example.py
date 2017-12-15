# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

from pyspark.sql.types import *
from pyspark.sql import types
from pyspark.sql import SQLContext
from pyspark.sql import HiveContext
from pyspark.sql import Row
from pyspark import SparkContext, SparkConf 

import pyspark.sql.functions
from pyspark.sql.functions import udf, array, sum

from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import OneHotEncoder, StringIndexer

from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import explode

import pyspark.mllib.linalg

import math

# /opt/spark/spark/bin/spark-submit   --jars `echo $(ls /opt/spark/spark/lib/*.jar) | tr ' ' ','`,`hbase classpath | tr ':' ','` --master local --driver-memory 4g --num-executors 4 --executor-memory 16g --executor-cores 2 `pwd`/ohe_example.py 


def cal_distance(pt_1, pt_2):
    print pt_1
    print pt_2
    print pt_1.squared_distance(pt_2)
    res = math.sqrt(pt_1.squared_distance(pt_2))
    print '-----------'
    return res


if __name__ == '__main__':
        
    conf = SparkConf().setAppName("One hot encoding")
    sc = SparkContext.getOrCreate(conf=conf)
    sqlContext = SQLContext(sc)
    cal_distance_udf = udf(cal_distance, DoubleType())

    # 如果資料長這樣
    # (0:"a"),
    # (1:"b"),
    # (2,"c"),
    # (3,"a"),
    # (4,"a"),
    # (5,"c", "a")

    # 要先拆成下面這樣
    # 而且裡面不得有空, 會壞掉, 一般要用會補NA的字串表示

    df = sqlContext.createDataFrame([
        ("id_0", "a"),
        ("id_1", "b"),
        ("id_2", "c"),
        ("id_3", "a"),
        ("id_4", "a"),
        ("id_5", "c"),
        ("id_5", "a")
    ], ["id", "category"])

    # one host encoding
    stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
    model = stringIndexer.fit(df)
    indexed = model.transform(df)
    encoder = OneHotEncoder(dropLast=False, inputCol="categoryIndex", outputCol="categoryVec")
    encoded = encoder.transform(indexed)
    encoded.show()
    encoded.printSchema()

    # +----+--------+-------------+-------------+
    # |  id|category|categoryIndex|  categoryVec|
    # +----+--------+-------------+-------------+
    # |id_0|       a|          0.0|(3,[0],[1.0])|
    # |id_1|       b|          2.0|(3,[2],[1.0])|
    # |id_2|       c|          1.0|(3,[1],[1.0])|
    # |id_3|       a|          0.0|(3,[0],[1.0])|
    # |id_4|       a|          0.0|(3,[0],[1.0])|
    # |id_5|       c|          1.0|(3,[1],[1.0])|
    # |id_5|       a|          0.0|(3,[0],[1.0])|
    # +----+--------+-------------+-------------+

    # root
    #  |-- id: string (nullable = true)
    #  |-- category: string (nullable = true)
    #  |-- categoryIndex: double (nullable = true)
    #  |-- categoryVec: vector (nullable = true)

    # categoryIndex表示category的編碼
    # categoryVec依次序表示是(總共幾類, [類別編碼], [值])
    # 如果要拿來算的話需要再轉成一般可以計算的向量
    # 就要用vectorAssembler

    # inputCols :欄位名稱的list(vector類別就可以, 也可以多個columns)
    # outputCol :輸出欄位名稱, 都可以

    # 把編碼後的原始資料丟進去, e.q.
    # df_features = vectorassembler.transform(encoded)


    vectorassembler = VectorAssembler(inputCols=['categoryVec'], outputCol='features')
    df_features = vectorassembler.transform(encoded)
    df_features.show()
    df_features.printSchema()

    # +----+--------+-------------+-------------+-------------+
    # |  id|category|categoryIndex|  categoryVec|     features|
    # +----+--------+-------------+-------------+-------------+
    # |id_0|       a|          0.0|(3,[0],[1.0])|[1.0,0.0,0.0]|
    # |id_1|       b|          2.0|(3,[2],[1.0])|[0.0,0.0,1.0]|
    # |id_2|       c|          1.0|(3,[1],[1.0])|[0.0,1.0,0.0]|
    # |id_3|       a|          0.0|(3,[0],[1.0])|[1.0,0.0,0.0]|
    # |id_4|       a|          0.0|(3,[0],[1.0])|[1.0,0.0,0.0]|
    # |id_5|       c|          1.0|(3,[1],[1.0])|[0.0,1.0,0.0]|
    # |id_5|       a|          0.0|(3,[0],[1.0])|[1.0,0.0,0.0]|
    # +----+--------+-------------+-------------+-------------+

    # root
    #  |-- id: string (nullable = true)
    #  |-- category: string (nullable = true)
    #  |-- categoryIndex: double (nullable = true)
    #  |-- categoryVec: vector (nullable = true)
    #  |-- features: vector (nullable = true)

    # 這裏還是會有同一個id多個category的狀況, 因為他們互斥 所以可以用reduce by key 把features向量相同id的加起來

    tmp_res = df_features.select("id","features").rdd.map(lambda x: ((x[0],),x[1])).reduceByKey(lambda x, y: x+y).map(lambda x:(x[0][0],x[1],)).toDF()
    tmp_res.show()
    total_id = tmp_res.count()
    # +----+-------------+
    # |  _1|           _2|
    # +----+-------------+
    # |id_4|[1.0,0.0,0.0]|
    # |id_5|[1.0,1.0,0.0]|
    # |id_2|[0.0,1.0,0.0]|
    # |id_3|[1.0,0.0,0.0]|
    # |id_0|[1.0,0.0,0.0]|
    # |id_1|[0.0,0.0,1.0]|
    # +----+-------------+

    all_vec = tmp_res.select("_2").rdd.flatMap(lambda r : r).collect()
    tmp2_res = tmp_res.rdd.map(lambda x: (x[0],x[1],all_vec,)).toDF().select("_1","_2",explode("_3"))
    total_pairwise = tmp2_res.count()
    tmp2_res.show(100)
    tmp2_res.printSchema()


    # 先取得所有的座標向量組成一個 list(all_vec) 然後把他接在每個座標點後面
    # +----+-------------+--------------------+
    # |  _1|           _2|                  _3|
    # +----+-------------+--------------------+
    # |id_4|[1.0,0.0,0.0]|[[1.0,0.0,0.0], [...|
    # |id_5|[1.0,1.0,0.0]|[[1.0,0.0,0.0], [...|
    # |id_2|[0.0,1.0,0.0]|[[1.0,0.0,0.0], [...|
    # |id_3|[1.0,0.0,0.0]|[[1.0,0.0,0.0], [...|
    # |id_0|[1.0,0.0,0.0]|[[1.0,0.0,0.0], [...|
    # |id_1|[0.0,0.0,1.0]|[[1.0,0.0,0.0], [...|
    # +----+-------------+--------------------+

    # 然後用explode 展開 '_3'就可以拿到兩兩點一組的pair
    # +----+-------------+-------------+
    # |  _1|           _2|          col|
    # +----+-------------+-------------+
    # |id_4|[1.0,0.0,0.0]|[1.0,0.0,0.0]|
    # |id_4|[1.0,0.0,0.0]|[1.0,1.0,0.0]|
    # |id_4|[1.0,0.0,0.0]|[0.0,1.0,0.0]|
    # |id_4|[1.0,0.0,0.0]|[1.0,0.0,0.0]|
    # |id_4|[1.0,0.0,0.0]|[1.0,0.0,0.0]|
    # |id_4|[1.0,0.0,0.0]|[0.0,0.0,1.0]|
    # |id_5|[1.0,1.0,0.0]|[1.0,0.0,0.0]|
    # |id_5|[1.0,1.0,0.0]|[1.0,1.0,0.0]|
    # |id_5|[1.0,1.0,0.0]|[0.0,1.0,0.0]|
    # |id_5|[1.0,1.0,0.0]|[1.0,0.0,0.0]|
    # |id_5|[1.0,1.0,0.0]|[1.0,0.0,0.0]|
    # |id_5|[1.0,1.0,0.0]|[0.0,0.0,1.0]|...


    tmp2_res = tmp2_res.withColumn('e_distance',cal_distance_udf(tmp2_res["_2"],tmp2_res["col"]))
    tmp2_res.show()

    # 然後就可以兩兩一組用udf算距離
    # +----+-------------+-------------+------------------+
    # |  _1|           _2|          col|        e_distance|
    # +----+-------------+-------------+------------------+
    # |id_4|[1.0,0.0,0.0]|[1.0,0.0,0.0]|               0.0|
    # |id_4|[1.0,0.0,0.0]|[1.0,1.0,0.0]|               1.0|
    # |id_4|[1.0,0.0,0.0]|[0.0,1.0,0.0]|1.4142135623730951|
    # |id_4|[1.0,0.0,0.0]|[1.0,0.0,0.0]|               0.0|
    # |id_4|[1.0,0.0,0.0]|[1.0,0.0,0.0]|               0.0|
    # |id_4|[1.0,0.0,0.0]|[0.0,0.0,1.0]|1.4142135623730951|
    # |id_5|[1.0,1.0,0.0]|[1.0,0.0,0.0]|               1.0|
    # |id_5|[1.0,1.0,0.0]|[1.0,1.0,0.0]|               0.0|
    # |id_5|[1.0,1.0,0.0]|[0.0,1.0,0.0]|               1.0|
    # |id_5|[1.0,1.0,0.0]|[1.0,0.0,0.0]|               1.0|
    # |id_5|[1.0,1.0,0.0]|[1.0,0.0,0.0]|               1.0|
    # |id_5|[1.0,1.0,0.0]|[0.0,0.0,1.0]|1.7320508075688772|
    # |id_2|[0.0,1.0,0.0]|[1.0,0.0,0.0]|1.4142135623730951|
    # |id_2|[0.0,1.0,0.0]|[1.0,1.0,0.0]|               1.0|
    # |id_2|[0.0,1.0,0.0]|[0.0,1.0,0.0]|               0.0|
    # |id_2|[0.0,1.0,0.0]|[1.0,0.0,0.0]|1.4142135623730951|
    # |id_2|[0.0,1.0,0.0]|[1.0,0.0,0.0]|1.4142135623730951|
    # |id_2|[0.0,1.0,0.0]|[0.0,0.0,1.0]|1.4142135623730951|
    # |id_3|[1.0,0.0,0.0]|[1.0,0.0,0.0]|               0.0|
    # |id_3|[1.0,0.0,0.0]|[1.0,1.0,0.0]|               1.0|
    # +----+-------------+-------------+------------------+

    tmp2_res_info = tmp2_res.describe(['e_distance'])
    min_dist = float(tmp2_res_info.rdd.filter(lambda x: x[0]=='min').collect()[0][1])
    max_dist = float(tmp2_res_info.rdd.filter(lambda x: x[0]=='max').collect()[0][1])
    if max_dist ==0:
        max_dist = 1

    # 用decribe把e_distane的最大最小值找出來

    # o x x
    # x o x
    # x x o
    # 現在拿到兩兩一組的距離, 可以表示成類似上面的矩陣, o 的部分是自己對自己, 距離是0, 不用管
    # x的部分算了兩次, 所以要除以2(total_distance)
    # 因為是兩兩一組, 所以全部組合數應該是把中間自己的扣掉再除以2(total_combination)
    # 平均各點距離就是用加起來的距離除以組合數

    total_distance = (tmp2_res.agg(sum('e_distance')).collect()[0][0])/2.0
    total_combination =  (total_pairwise - total_id)/2.0
    print total_distance    
    avg_distance = total_distance/(total_combination)

    print "min distance is ", min_dist
    print "max distance is ", max_dist
    print 'average distance is',avg_distance

    # 再做最大最小正規化    
    normalized_avg_dist = (avg_distance-min_dist)*1.0/(max_dist-min_dist)
    print 'After min-max normalization is ',normalized_avg_dist




