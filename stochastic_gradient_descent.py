import numpy as np
import time
import math
import random


def stochastic_gradient_descent(X,n,y,len_w,num_epochs,current_epoch,w,learning_rate):
    #randomly map all data points to a new order (since m = 1)
    deck = list(range(0, n))
    random.shuffle(deck)
    map_data_to_parts = [deck.pop() for i in range(n)] #data part i = map_data_to_parts[i]

    new_w = np.empty(len_w)
    wT = w.transpose()

    for i in range(n): #since m=1, there are n random parts D_i
        data_point_i = map_data_to_parts[i]
        y_p = y[data_point_i]
        x_p = X[data_point_i]
        y_p_hat = np.dot(wT,x_p)

        for j in range(len_w):
            new_w[j] = w[j] + learning_rate*(y_p - y_p_hat)*x_p[j] #since m = 1, there is only one x_p in D_i

    if(current_epoch==num_epochs):
        return new_w
    print(time.time())
    return stochastic_gradient_descent(X,n,y,len_w,num_epochs,current_epoch+1,new_w,learning_rate)

def calc_loss_value(X,y,w,n):
    sum = 0
    wT = w.transpose()
    for i in range(n):
        wTxi = np.dot(wT,X[i])
        sum = sum + math.pow(y[i] - wTxi,2)
    sum = sum/(2*n)
    return sum

def print_to_output(w):
    with open('assign2q3b_output.tsv', 'w') as f:
        #headers
        for i in range(len(w)-1):
            f.write("w"+str(i+1)+"\t")
        f.write("w0")
        f.write("\n")

        #weights
        w0 = w[-1]
        for wi in w[:-1]:
            f.write(str(wi)+"\t")
        f.write(str(w0))

start = time.time()

with open('pa2-data/data_100k_300.tsv', 'r') as f:
    lines = [line.rstrip('\n').split() for line in f]
    n = int(lines[0][0])
    num_features = int(lines[1][0])
    X = np.array([line[1:] for line in lines[3:]]) #features
    y = np.array([line[0] for line in lines[3:]]) #labels
    X = X.astype(np.float)
    y = y.astype(np.float)


#add bias term
biases = np.ones(shape=(len(X),1))
X = np.append(X, biases, axis=1)
len_w = num_features + 1  # +1 due to added bias term

num_epochs = 12
learning_rate = 0.0000001
initial_w = np.random.random_sample(len_w)

w = stochastic_gradient_descent(X,n,y,len_w,num_epochs,1,initial_w,learning_rate) #1 = current epoch number
print_to_output(w)

loss = calc_loss_value(X,y,w,n)
print("loss function value: "+str('%.2E' % loss))

end = time.time()
print("elapsed time: "+str((end-start)/60.0)+" minutes")
