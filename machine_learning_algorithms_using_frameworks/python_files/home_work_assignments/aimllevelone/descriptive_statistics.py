
from collections import Counter
import math
import matplotlib.pyplot as plt

mydata = [ 945, 767, 911, 823, 701, 714, 767, 945, 833, 402, 333, 215, 900, 899, 677, 567, 666, 979, 857, 979, 979, 455, 699, 1355, 215, 979, 833, 823, 979, 1399, 1455, 1297, 375]
no_of_items = len(mydata)
print("The given list is as below: \n")
print(mydata)
print("\n")
print("Number of items in the list: ", no_of_items)
min = mydata[0]
max = mydata[0]
sum = 0

#### Finding the Minimum & Maximum numbers
for i in range(no_of_items):
    sum += mydata[i]
    if mydata[i] < min:
        min = mydata[i]
    if mydata[i] > max:
        max = mydata[i]
print("The minimum number is: ", min)
print("The maximum number is: ", max)
print("\n")
#### Mean, Median & Mode
mean = sum/no_of_items
print("The mean is: ", mean)
mydata.sort()
#print(mydata)
if no_of_items % 2 == 0:
    median1 = mydata[no_of_items//2]
    median2 = mydata[no_of_items//2 - 1]
    median = (median1 + median2)/2
else:
    median = mydata[no_of_items//2]
print("The Median is: ", median)

dict = {}
highestNum = 0
length = len(mydata)
for i in range(length):
    dict.update({mydata[i]:mydata.count(mydata[i])})
for i in dict.keys():
    if dict[i] > highestNum:
        highestNum = dict[i]
        mode = i
if highestNum != 1: print("The Mode is: ", mode)
elif highestNum == 1: print("All elements of list appear once.")
print("\n")
#### Variance & Standard Deviation
variance = 0
std_dev = 0
var1 = 0
for i in range(no_of_items):
    var1 += pow((mydata[i] - mean), 2)
variance = var1/no_of_items
std_dev = math.sqrt(variance)
print("The variance is: ", variance)
print("The standard deviation is: ", std_dev)
print("\n")
#### 1st, 2nd & 3rd quartiles
list1 = []
list2 = []
for i in range(length):
    if mydata[i] < median:
        list1.append(mydata[i])
    elif mydata[i] > median:
        list2.append(mydata[i])
if len(list1) % 2 == 0:
    m1 = list1[len(list1)//2]
    m2 = list1[len(list1)//2 - 1]
    print("The 1st Quartile is: ", (m1 + m2)/2)
else:
    print("The 1st Quartile is: ", list1[len(list1)//2])
print("The 2nd Quartile is: ", median)
if len(list2) % 2 == 0:
    m1 = list2[len(list2)//2]
    m2 = list2[len(list2)//2 - 1]
    print("The 3rd Quartile is: ", (m1 + m2)/2)
else:
    print("The 3rd Quartile is: ", list2[len(list2)//2])

plt.boxplot(mydata)
plt.show()
