import numpy as np

test_set=np.random.rand(3,5)
training_set=np.random.rand(10,5)
n_test=test_set.shape[0]
n_train=training_set.shape[0]
#the data has dimmention of 5, there are 3 test data and 10 training data

dists=np.zeros((n_test,n_train))
for i in range(n_test):
    for j in range(n_train):
        dists[i][j]=np.sqrt(np.sum(np.square(training_set[j]-test_set[i])))
print(dists)

dists=np.zeros((n_test,n_train))
for i in range(n_test):
    square=np.square(training_set-test_set[i])
    square_sum=np.sum(square,axis=1)
    dists[i]=np.sqrt(square_sum)
print(dists)
'''
print(test_set[1])
print(training_set-test_set[1])
you can see that the subduction is along the row vector
'''
cross=np.dot(test_set,training_set.T)
test_square=np.square(test_set)
training_square=np.square(training_set)

#pls notice the shape of np.sum(array,axix=n)
#it would result in a list instead of an array
#in order to make it an array, you need to add axis to it according to the desired outcome
test_square_sum=np.sum(test_square,axis=1)[:,np.newaxis]
training_square_sum=np.sum(training_square,axis=1)[np.newaxis,:]
square=test_square_sum+training_square_sum
final=square-2*cross

dists = np.sqrt(final)
print(dists)
