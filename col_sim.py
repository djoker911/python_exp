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
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix, CoordinateMatrix
from py4j.java_gateway import JavaObject

from pyspark import RDD
from pyspark.mllib.common import callMLlibFunc, JavaModelWrapper
from pyspark.mllib.linalg import _convert_to_vector, Matrix

# /opt/spark/spark/bin/spark-submit   --jars `echo $(ls /opt/spark/spark/lib/*.jar) | tr ' ' ','`,`hbase classpath | tr ':' ','` --master local --driver-memory 4g --num-executors 4 --executor-memory 16g --executor-cores 2 `pwd`/col_sim.py 

class NewVersionIndexedRowMatrix(IndexedRowMatrix):
    
    def yaha(self):
        print "yaaaaaaaaaaaaaaaaaaaaa"

    def columnSimilarities(self):
        """
        Compute all cosine similarities between columns.
        >>> rows = sc.parallelize([IndexedRow(0, [1, 2, 3]),
        ...                        IndexedRow(6, [4, 5, 6])])
        >>> mat = IndexedRowMatrix(rows)
        >>> cs = mat.columnSimilarities()
        >>> print(cs.numCols())
        3
        """
        java_coordinate_matrix = self._java_matrix_wrapper.call("columnSimilarities")
        return CoordinateMatrix(java_coordinate_matrix)


if __name__ == '__main__':
        
    conf = SparkConf().setAppName("Column Similarities")
    sc = SparkContext.getOrCreate(conf=conf)
    sqlContext = SQLContext(sc)

    rows = sc.parallelize([IndexedRow(0, [1, 2, 3]),IndexedRow(1, [4, 5, 6])])
    mat = NewVersionIndexedRowMatrix(rows)
    mat.yaha()
    # Get its size.
    m = mat.numRows()
    n = mat.numCols()
    print m
    print n
    cs = mat.columnSimilarities()
    print cs.entries.max() #MatrixEntry(0, 1, 0.990830168044)
    print cs.entries.max().value # 0.990830168044
    print cs.entries.map(lambda x: x.value).mean() #mean of value
