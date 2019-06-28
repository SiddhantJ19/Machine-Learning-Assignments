import random
import matplotlib.pyplot as plt
from math import exp

def predict_high(a):
    a1 = [[0] for i in a]
    a1[a.index(max(a))] = [1.0]
    return a1

def compare(a, b):
    if a.index([1.0]) == b.index([1.0]):
        return 0
    return 1

def first_element(a, classes):
    for i in range(len(a)):
        a[i] = [[0] if j+1 !=a[i][0] else [1] for j in range(classes)]
    return a

def multiply(X,Y):
    return [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)] for X_row in X]

def dot_product(a, b):
    l = [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]
    return l

def subtract(a,b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def transpose(a):
    b = [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]
    return b

def division(a,b):
    assert not isinstance(b, list)
    return [[a[i][j] / b for j in range(len(a[0]))] for i in range(len(a))]

def sigmoid(a):
    return [[1/(1 + exp(-a[i][j])) for j in range(len(a[0]))] for i in range(len(a))]

def propogation(X, W1, W2):
    X_hidden = multiply(W1, X)
    A_hidden = sigmoid(X_hidden)    
    A_hidden = [[1.0]] + A_hidden
    X_output = multiply(W2, A_hidden)
    A_output = sigmoid(X_output)
    return A_output, [X_output, A_hidden]


file = open('Colon_Cancer_CNN_Features.csv','r')
f = file.readlines()
f = [list(map(float , i.split(',')))  for i in f]
f = list(filter(lambda line : bool(line), f))
file.close()
random.shuffle(f)

f = [[1.0] + x for x in f]
length = len(f)
examples = int(0.8 * length)

f1 = f[0 : examples]

train_x = [ x[0 : len(f1[0])-1]  for x in f1]
train_x = [[ [train_x[i][j]] for j in range(len(train_x[0])) ] for i in range(len(train_x))]

train_y = [ y[len(f1[0])-1 : len(f1[0])]  for y in f1]

features = len(train_x[0])

learning_rate = 0.001
classes = 4
hidden_layer_range = range(5,16)

test_accuracy = []

for hidden_layer in hidden_layer_range:
    W1 = [[random.uniform(-1,1) for i in range(features)] for j in range(hidden_layer)]
    W2 = [[random.uniform(-1,1) for i in range(hidden_layer+1)] for j in range(classes)] 
    a = first_element(train_y, classes)
    epochs = 5
    for o in range(epochs):
        for i in range(examples):
            z, cache = propogation(train_x[i], W1, W2)
            dz = subtract(z, a[i])
            ones = [[1] for j in range(classes)]
            dnet = dot_product(z, subtract(ones,z))

            first_dot = dot_product(dz, dnet)
            del2 = multiply(first_dot, transpose(cache[1]))
            term1 = multiply(transpose(W2), first_dot)
            ones = [[1] for j in range(len(cache[1]))]
            term2 = dot_product(cache[1], subtract(ones, cache[1]))
            second_dot = dot_product(term1, term2)
            second_dot = second_dot[1:]
            del1 = multiply(second_dot, transpose(train_x[i]))


            # W2
            W2 = subtract(W2, division(del2, 1/learning_rate))
            # W1
            W1 = subtract(W1, division(del1, 1/learning_rate))

    
    f2 = f[examples : ]
    test_x = [ x[0 : len(f2[0])-1]  for x in f2]
    test_y = [ y[len(f2[0])-1 : len(f2[0])]  for y in f2]

    test_x = [[ [test_x[i][j]] for j in range(len(test_x[0])) ] for i in range(len(test_x))]
    encoded_y = first_element(test_y, classes) 

    total_cost = 0
    test_examples = len(f2)
    for t in range(test_examples):
        pred_val,_ = propogation(test_x[t], W1, W2)    
        misclassified = compare(predict_high(pred_val), encoded_y[t])
        total_cost += misclassified 
    
    store_array = (total_cost*100)/test_examples
    test_accuracy.append(store_array)


    print("Accuracy over test set with {} neurons in hidden layer is: {}".format(hidden_layer, store_array))

