import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def preprocess_dataset(X):
    m = np.mean(X, axis=0)
    m = m*np.ones(X.shape)
    s = np.std(X, axis=0)
    out = (X-m)/s
    return out, m, s

df = pd.read_csv('datasets/BostonHousing.csv')
dataset = df.values.astype('float32')

dataset, m, s = preprocess_dataset(dataset)

# split into input (X) and output (Y) variables
X = dataset[:,0:13]
y = dataset[:,13]

print(X)

seed = 15
np.random.seed(seed)


def preprocess_input(X, m, s):
    return (X-m)/s

def reconstruct_output(y, m, s):
    return y*s+m

# create training and testing vars
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Creating validation and training data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
print(X_train.shape, y_train.shape, "-->  Train samples")
print(X_val.shape, y_val.shape, "-->  Validation samples")
print(X_test.shape, y_test.shape, "--> Test samples")

# number_hidden_layers
H = 1000
C = 0.01
N = X_train.shape[0]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def output_hidden_layer(x, Win):
    aux = np.ones(x.shape[0])
    aux = aux.reshape((aux.shape[0], 1))
    x = np.concatenate((x, aux), axis=1)
    out = np.dot(x, Win)
    #return sigmoid(out)
    return np.maximum(out, 0, out) # ReLu activation function

def predict(x, Win, V):
    x = output_hidden_layer(x, Win)
    y = np.dot(x, V)
    return y

val_err = []
train_err = []
C_train = []
C_val = []
C_list = [2**c for c in range(-24, 26)]
C = 0.5

for H in range(1, 301):# 20001):
    # weights of hidden layer
    Win = np.random.normal(size=[X_train.shape[1]+1, H])

    Z = output_hidden_layer(X_train, Win)
    Zt = np.transpose(Z)


    if H <= N:
        V = np.dot(np.dot(np.linalg.inv(np.eye(Z.shape[1], dtype='float32')/C + np.dot(Zt, Z)), Zt), y_train.reshape((y_train.shape[0], 1)))
    elif H > N:
        V = np.dot(np.dot(Zt, np.linalg.inv(np.eye(Z.shape[0], dtype='float32')/C + np.dot(Z, Zt))), y_train.reshape((y_train.shape[0], 1)))

    ypt = predict(X_train, Win, V)
    ypv = predict(X_val, Win, V)

    y_train = y_train.reshape((y_train.shape[0], 1))
    y_val = y_val.reshape((y_val.shape[0], 1))

    # Least mean square error
    E_train = np.sum((y_train - ypt)**2)
    E_train /= y_train.shape[0]
    train_err.append(E_train)

    E_val = np.sum((y_val - ypv)**2)
    E_val /= y_val.shape[0]
    val_err.append(E_val)




train_err = np.array(train_err)
val_err = np.array(val_err)

print(train_err)
print(val_err)

t_min = np.argmin(train_err)
v_min = np.argmin(val_err)

diff = np.abs(train_err - val_err)
d_min = np.argmin(diff)

print('Number of neurons in hidden layer with minimum train error: ', t_min)
print('Number of neurons in hidden layer with minimum val error: ', v_min)
print('Number of neurons in hidden layer with minimum difference between val and train errors: ', d_min)

plt.figure(1)
plt.plot(train_err, 'g')
plt.plot(val_err, 'r')
plt.title('Least Mean Square Error')
plt.ylabel('error')
plt.xlabel('number of neurons in hidden layer')
plt.legend(['train', 'validation'], loc='upper left')

# plt.figure(2)
# plt.plot(C_train, 'g')
# plt.plot(C_val, 'r')
# plt.title('Least Mean Square Error - 10000 neurons in hidden layer')
# plt.ylabel('error')
# plt.xlabel('parameter C')
# plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#plt.plot(train_err, 'r',val_err, 'g')
#plt.show()






