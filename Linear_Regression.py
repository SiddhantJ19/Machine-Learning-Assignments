
# coding: utf-8

# In[102]:


from random import shuffle
from matplotlib import pyplot as plt
n = 0.03
file = open('Desktop/yacht_hydrodynamics.txt','r')
vector=[]
data = file.read()
data = data.split('\n')
for line in data:
    vector.append(line.split())
vector.pop(309)
vector.pop(308)

for gg in range(len(vector)):
    vector[gg] = [float(i) for i in vector[gg]]


# In[103]:


for element in vector:
    element.insert(0,1.0)


# In[104]:


X = []
Y = []
shuffle(vector)
for element in vector:
    X.append(element[0:7])
    Y.append(element[7])


# In[105]:


X_train , X_test = X[0:200],X[200:]
Y_train , y_test = Y[0:200],Y[200:]
w = [1.0 ,1.0, 1.0 ,1.0 ,1.0, 1.0, 1.0]
pred = []


# In[106]:


def dotproduct(A,B):
    return sum([a*b for a,b in zip(A,B)])

def diff(w):
    pred = [ dotproduct(row,w) for row in X_train ]
    pred = [ Y_train[i] - pred[i] for i in range(len(pred))]
    return pred


# In[107]:


def change(differ,j):
    '''
    gradient at index j. W'[j] = W[j] + n*change[j]
    '''
    return sum([ differ[item] * X_train[item][j] for item in range(len(differ))])/len(differ)


# In[108]:


def Error(X_train,W,Y_train):
    return ( sum( [0.5*(dotproduct(X_train[i],W)-Y_train[i])**2 for i in range(len(X_train))] ) )/len(X_train)


# In[109]:


allErrors=[]

def gradient(w):
    t = 50
    while(n*t>0.001):
        differ=diff(w)
        for j in range(len(w)):
            t = change(differ,j)
            w[j] = w[j] + n*t
        allErrors.append(Error(X_train,w,Y_train))
    plt.plot(allErrors)
    plt.show()
    return w


# In[110]:


w=gradient(w)
print(w)


# In[111]:


print(Error(X_train,w,Y_train))


# In[112]:


ss= sum( [ (dotproduct(X_train[i],w)- Y_train[i])**2 for i in range(len(X_train))] )
mean = sum(Y_train)/len(Y_train)
MSE = sum( [ (mean - y_tr)**2 for y_tr in Y_train])
print(1-(ss/MSE))


# In[113]:


ss= sum( [ (dotproduct(X_test[i],w)- y_test[i])**2 for i in range(len(X_test))] )
mean = sum(y_test)/len(y_test)
MSE = sum( [ (mean - y_tr)**2 for y_tr in y_test])
print(1-(ss/MSE))

