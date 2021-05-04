# Author: Manohar Mukku
# Date: 23.08.2018
# Desc: Apriori Algorithm to find frequent itemsets
# Link: https://github.com/manoharmukku/data-mining-projects/apriori

import sys
import pandas as pd
import itertools

def find_frequent_1_itemsets(data, min_sup):
    # Generate Candidate 1-itemsets
    C0 = {}
    for transaction in data:
        for item in transaction:
            if (item in C0):
                C0[item] += 1
            else:
                C0[item] = 1

    # Find the frequent 1-itemsets
    L0 = []
    for key, value in C0.items():
        if (value >= min_sup):
            L0.append([[key], value])

    # Return the frequent 1-itemset
    return sorted(L0)

def find_frequent_k_itemsets(Ck, data, min_sup):
    # Get the counts of itemsets in Ck
    Ck_dict = {}
    for i in range(len(Ck)):
        for transaction in data:
            if (set(Ck[i]) <= set(transaction)):
                if (tuple(Ck[i]) in Ck_dict):
                    Ck_dict[tuple(Ck[i])] += 1
                else:
                    Ck_dict[tuple(Ck[i])] = 1

    # Find the frequent itemsests in Ck
    Lk = []
    for key, value in Ck_dict.items():
        if (value >= min_sup):
            Lk.append([list(key), value])

    # Return the sorted frequent k-itemset
    return sorted(Lk)

def check_and_join(l1, l2):
    c = []
    k = len(l1)

    if (k == 1):
        c.extend(l1)
        c.extend(l2)
        return c

    flag = True
    for i in range(k-1):
        if (l1[i] != l2[i]):
            flag = False
            break

    if (flag):
        c.extend(l1[0:k-2])

        if (l1[k-1] < l2[k-1]):
            c.extend(l1[k-1])
            c.extend(l2[k-1])
        else:
            c.extend(l2[k-1])
            c.extend(l1[k-1])

    return c

def has_infrequent_subset(c, l_prev):
    # Create a dictionary of l_prev
    l_prev_dict = {}
    for i in range(len(l_prev)):
        l_prev_dict[tuple(l_prev[i][0])] = 1

    # Iterate over all subsets of c and check whether they are all present in l_prev
    k = len(l_prev[0][0])
    for subset in itertools.combinations(c, k):
        if (subset not in l_prev_dict):
            return True

    return False

def apriori_gen(l_prev):
    # Construct candidate k-itemset, Ck by joining and pruning itemsets in l_prev
    Ck = []
    for i in range(len(l_prev)-1):
        for j in range(i+1, len(l_prev)):
            c = check_and_join(l_prev[i][0], l_prev[j][0])
            if (len(c) > 0 and has_infrequent_subset(c, l_prev) == False):
                Ck.append(c)

    return Ck

# Sanity check of command line arguments
if (len(sys.argv) != 4):
    print ("Usage: python main.py minimum_support_count_threshold min_confidence data_file_path")
    sys.exit()

# Mimimum support count threshold
min_sup = int(sys.argv[1])
if (min_sup <= 0):
    print ("Error: Minimum support count value should be positive")
    sys.exit()

min_conf = int(sys.argv[2])
if (min_conf <= 0):
    print ("Error: Minimum confidence value should be positive")
    sys.exit()

# Read the csv data file to a pandas dataframe
df = pd.read_csv(sys.argv[3], header=None, names=['List of items'])

# Store the dataframe as a list of lists
data = []
for _, row in df.iterrows():
    data.append(sorted(row['List of items'].split(",")))

# Find frequent 1-itemsets
L0 = find_frequent_1_itemsets(data, min_sup)

# Append L0 to the final frequent itemsets list
L = []
L.append(L0)

# Iterate on k to find frequent k-itemsets
k = 1
while (len(L[k-1]) > 0):
    Ck = apriori_gen(L[k-1])
    Lk = find_frequent_k_itemsets(Ck, data, min_sup)
    L.append(Lk)
    k += 1

# Print the frequent itemsets
print ('{:30}{:15}'.format('Frequent Itemset', 'Support Count'))
print ("--------------------------------------------")
for freq_itemset in L:
    for itemset in freq_itemset:
        print ('{:30}{:5}'.format(str(itemset[0]), itemset[1]))

# Store the support counts in dictionary
sup_count = {}
for freq_itemset in L:
    for itemset in freq_itemset:
        sup_count[tuple(itemset[0])] = itemset[1]

# Find and print the association rules
print ("\nAssociation rules:")
print ("--------------------")
for freq_itemset in L:
    for itemset in freq_itemset:
        if (len(itemset[0]) <= 1):
            continue

        # Generate all the subsets of the current itemset
        for i in range(1,len(itemset[0])):
            for subset in itertools.combinations(itemset[0], i):
                # Check the association rule confidence
                if ((sup_count[subset] / sup_count[tuple(sorted(set(itemset[0]) - set(subset)))]) >= min_conf):
                    print (subset,' => ', tuple(sorted(set(itemset[0]) - set(subset))))
