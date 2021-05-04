# Sample code to do Eclat in Python

# The package source can be found at below location, it is not stable however
# https://github.com/andi611/Apriori-and-Eclat-Frequent-Itemset-Mining
from association import eclat

# Creating Sample Transactions
transactions = [
    ['Milk', 'Bread', 'Saffron'],
    ['Milk', 'Saffron'],
    ['Bread', 'Saffron','Wafer'],
    ['Bread','Wafer'],
 ]

# Creating Rules using Eclat
Rules=eclat.eclat(data=transactions, min_support=0.5)
print(Rules)