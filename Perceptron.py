
# coding: utf-8

# In[65]:


from random import shuffle
from matplotlib import pyplot as plt
n = 0.01
file = open('Desktop/iris.data','r')
vector=[]
data = file.read()
data = data.split('\n')
for line in data:
    vector.append(line.split(','))

vector = [ v[:-1]+[ '1' if 'virginica' in v[-1] else '-1'] for v in vector if 'setosa' not in v[-1] and len(v)>1]


for gg in range(len(vector)):
    vector[gg] = [float(i) for i in vector[gg]]


# In[66]:


for element in vector:
    element.insert(0,1.0)


# In[67]:


X = []
Y = []
shuffle(vector)
for element in vector:
    X.append(element[0:-1])
    Y.append(element[-1])



# In[68]:


X_train , X_test = X[0:70],X[70:]
Y_train , y_test = Y[0:70],Y[70:]
w = [1.0 ,1.0, 1.0 ,1.0 ,1.0]
pred = []


# In[69]:


def dotproduct(A,B):
    return sum([a*b for a,b in zip(A,B)])

def diff(w):
    pred = [ dotproduct(row,w) for row in X_train ]
    pred = [ Y_train[i] - ( 1 if pred[i]>0 else -1) for i in range(len(pred))]
    return pred


# In[70]:


def change(differ,j):
    '''
    gradient at index j. W'[j] = W[j] + n*change[j]
    '''
    return sum([ differ[item] * X_train[item][j] for item in range(len(differ))])/len(differ)


# In[71]:


def Error(X_train,W,Y_train):
    misclassified=0
    for i in range(len(X_train)):
        if (dotproduct(X_train[i],W) * Y_train[i]<0):
            misclassified+=1
    return misclassified / len(X_train)


# In[72]:


allErrors=[]
def gradient(w):
    for i in range(15000):
        differ=diff(w)
        w = [ w[i] + n * change(differ,i) for i in range(len(w)) ]
        allErrors.append(Error(X_train,w,Y_train))
    return w


# In[73]:


w=gradient(w)
print(w)


# In[74]:


print(Error(X_train,w,Y_train))


# In[75]:


ss= sum( [ (dotproduct(X_train[i],w)- Y_train[i])**2 for i in range(len(X_train))] )
mean = sum(Y_train)/len(Y_train)
MSE = sum( [ (mean - y_tr)**2 for y_tr in Y_train])
print(1-(ss/MSE))


# In[76]:


ss= sum( [ (dotproduct(X_test[i],w)- y_test[i])**2 for i in range(len(X_test))] )
mean = sum(y_test)/len(y_test)
MSE = sum( [ (mean - y_tr)**2 for y_tr in y_test])
print(1-(ss/MSE))

