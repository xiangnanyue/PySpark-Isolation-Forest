# PySpark-Isolation-Forest
A tested version of pyspark isolation forest

The User needs to define his or her own metadata.json and feature file in order to use the python script.

Steps of coding the pyspark isolation forest can be found in the notebook.

If files have been defined, you can simply run 

>>spark-submit --master local[2] --py-files Node.py IsolationForestSpark.py --filePath $fileNameInput --nb_trees 100 --nb_samples 256

