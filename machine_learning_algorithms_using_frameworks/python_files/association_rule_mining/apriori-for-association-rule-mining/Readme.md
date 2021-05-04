# Implementation of Apriori Algorithm

This implementation of Apriori algorithm finds the frequent itemsets from a given set of transactions and for a given value of minimum support count threshold and also finds the Association rules satisfying the minimum values of support count and confidence.
__Python 3.6.5__ was used for this implementation.

### How to Run:
First clone the project:
```
$ git clone https://github.com/manoharmukku/data-mining-projects/apriori
```
Then go the cloned directory and run with the desired parameters as below:
```
$ python main.py min_support_count_threshold min_confidence data_file_path
```
For example:
```
$ python main.py 2 1 data/GroceryStoreDataSet.csv
```

__DataSet Used:__ https://www.kaggle.com/ankitkatiyar91/apriori-algorithm/data

###### References:
* Chapter 6 from _Data Mining by Han & Kamber, 3rd Edition, Morgan Kuffman_
* https://stackoverflow.com/questions/7378180/generate-all-subsets-of-size-k-containing-k-elements-in-python
* https://stackoverflow.com/questions/16579085/python-verifying-if-one-list-is-a-subset-of-the-other
* https://pyformat.info
* https://stackoverflow.com/questions/473099/check-if-a-given-key-already-exists-in-a-dictionary-and-increment-it
* https://www.quora.com/How-does-one-hash-lists-in-Python
* https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
* https://stackoverflow.com/questions/7844118/how-to-convert-comma-delimited-string-to-list-in-python
* https://stackoverflow.com/questions/36139/how-to-sort-a-list-of-strings
* https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
