# python_exp
Code for experiments


ohe_example.py
	one hot encoding by pyspark, and calculaing average distance of pairwise points

col_sim.py
	New an object which inherit IndexedRowMatrix, and add columnSimilarities, cuz
	this method is not in spark 1.6.1, but it has been released in later branch
