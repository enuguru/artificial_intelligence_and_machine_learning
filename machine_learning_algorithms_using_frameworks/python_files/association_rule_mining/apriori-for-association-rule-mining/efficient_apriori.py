
from efficient_apriori import apriori
transactions = [('eggs', 'bacon', 'soup'),
                        ('eggs', 'bacon', 'apple'),
                                        ('soup', 'bacon', 'banana')]
itemsets, rules = apriori(transactions, min_support=0.5, min_confidence=1)
print(rules)  # [{eggs} -> {bacon}, {soup} -> {bacon}]
