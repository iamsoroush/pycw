# pycw
Python implementation of [Chinese-Whispers](https://www.researchgate.net/publication/228670574_Chinese_whispers_An_efficient_graph_clustering_algorithm_and_its_application_to_natural_language_processing_problems) clustering algorithm, compatible with [scikit-learn](https://scikit-learn.org/).


The main strength of Chinese Whispers lies in its time linear property. This algorithm works fine for complex clustering problems such as face clustering, which have a high dimensionality in representation space and consists of an unknown number of clusters. Furthermore, it relies only on two parameters which can easily optimised.


# Requirements:
```
1. numpy
2. scikit-learn
```


# Usage
Just like a sklearn estimator, do the following:
```python
from pycw import ChineseWhispers

es = ChineseWhispers(n_iteration=2, metric='euclidean')
predicted_labels = es.fit_predict(X)
```

A simple usage example can be found in the [provided notebook](https://github.com/iamsoroush/pycw/blob/master/pycw_example.ipynb). Here is the performance of the **_pycw_** on a set of artificially generated data:

![alt text](https://github.com/iamsoroush/pycw/blob/master/index.png "Ground truth")
![alt text](https://github.com/iamsoroush/pycw/blob/master/index1.png "pycw results")
