import sys

input_file = open('feature_vectors.txt', 'r')
output_file = open('feature_vectors_out.txt', 'w')

for line in input_file:
    items = line.strip().split()
    feature_vector = []
    feature_vector.append(items[0])
    for j in range(3,len(items)):
        feature_vector.append(items[j])
#   Map converts list to string, joins elements
    print(' '.join(map(str,feature_vector)), file=output_file)

input_file.close()
output_file.close()

