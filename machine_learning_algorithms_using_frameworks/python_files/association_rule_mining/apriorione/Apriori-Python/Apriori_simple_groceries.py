#!/usr/bin/env python
#The dict {} in python such as:my_information = {'name': 'Pusheen the Cat', 'country': 'USA', 'favorite_numbers': [42, 105]}
#name -> Pusheen the Cat (Key is name , value is Pusheen the cat)
def Apriori_gen(Itemset, lenght):
    """Too generate new (k+1)-itemsets can see README Join Stage"""
    canditate = []
    canditate_index = 0
    for i in range (0,lenght):
        element = str(Itemset[i])
        for j in range (i+1,lenght):
            element1 = str(Itemset[j])
            if element[0:(len(element)-1)] == element1[0:(len(element1)-1)]:
                    unionset = element[0:(len(element)-1)]+element1[len(element1)-1]+element[len(element)-1] #Combine (k-1)-Itemset to k-Itemset 
                    unionset = ''.join(sorted(unionset))  #Sort itemset by dict order
                    canditate.append(unionset)
    print(canditate)
    return canditate

def Apriori_prune(Ck,MinSupport):
    L = []
    for i in Ck:
        if Ck[i] >= minsupport:
            L.append(i)
    return sorted(L)
def Apriori_count_subset(Canditate,Canditate_len):
    """ Use bool to know is subset or not """
    Lk = dict()
    file = open('simple_groceries.csv')
    for l in file:
        l = str(l.split())
        count = 0
        for i in range (0,Canditate_len):
            key = str(Canditate[i])
            if key not in Lk:
                Lk[key] = 0
            flag = True
            for k in key:
                if k not in l:
                    flag = False
            if flag:
                Lk[key] += 1
    file.close()
    return Lk
minsupport = 2
C1={} 
file = open('simple_groceries.csv')
"""Count one canditate"""
for line in file:
    for item in line.split():
        if item in C1:
            C1[item] +=1
        else:
            C1[item] = 1
file.close()
sorted(C1.keys())
L = []
L1 = Apriori_prune(C1,minsupport)
L = Apriori_gen(L1,len(L1))
print(L)
print('====================================')
print('Frequent 1-itemset is',L1)
print('====================================')
k=2
while L != []:
    C = dict()
    C = Apriori_count_subset(L,len(L))
    fruquent_itemset = []
    fruquent_itemset = Apriori_prune(C,minsupport)
    print('====================================')
    print('Frequent',k,'-itemset is',fruquent_itemset)
    print('====================================')
    L = Apriori_gen(fruquent_itemset,len(fruquent_itemset))
    k += 1
