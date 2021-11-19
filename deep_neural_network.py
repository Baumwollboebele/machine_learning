
from abc import ABCMeta
import numpy as np

def foo(x):
    return x^3+x^2+2

def relu(x):
    return np.maximum(0,x)


x = np.arange(-5,5,1)
y = list(map(foo,x))

INPUT_SIZE=1
HIDDEN_LAYER_0_SIZE = 5
HIDDEN_LAYER_1_SIZE = 5
OUTPUT_SIZE = 1

seed = 42


B1 = np.full((1,HIDDEN_LAYER_0_SIZE),0.1)
B2 = np.full((1,HIDDEN_LAYER_1_SIZE),0.1)
B_OUT = np.full((1,OUTPUT_SIZE),0.1)

W_H1 = np.random.rand(INPUT_SIZE,HIDDEN_LAYER_0_SIZE)
W_H2= np.random.rand(HIDDEN_LAYER_1_SIZE,HIDDEN_LAYER_0_SIZE)
W_OUT = np.random.rand(HIDDEN_LAYER_1_SIZE,OUTPUT_SIZE)

def feed_forward(X):
    A0 = np.dot(X,W_H1) + B1
    Z0 = relu(A0)
    print(f"Z0: {Z0}")
    
    A1 = np.dot(Z0,W_H2) + B2
    Z1 = relu(A1)
    print(f"Z1: {Z1}")

    A_OUT = np.dot(Z1,W_OUT) + B_OUT
    Z_OUT = relu(A_OUT)
    print(f"Z_OUT: {Z_OUT}")

    return Z_OUT

feed_forward(x[0])

