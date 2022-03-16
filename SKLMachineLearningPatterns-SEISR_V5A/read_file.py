import numpy as np

bins = []
prob = []

fname = 'probability_file.txt'
data_prob = open(fname,'r')

for line in data_prob:
    x=line.split()
    bins.append(x[0])
    prob.append(x[1])
    
print(bins[:])
print(prob[:])

data_prob.close()