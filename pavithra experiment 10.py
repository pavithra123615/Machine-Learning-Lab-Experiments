import numpy as np

x = np.array([0.5, 1.2])
y = 1.0
w = np.array([0.1, 0.2]); b = 0.0
lr = 0.1

def sigmoid(z): return 1/(1+np.exp(-z))

z = np.dot(w, x) + b
a = sigmoid(z)
loss = -(y*np.log(a) + (1-y)*np.log(1-a))

dz = a - y
dw = dz * x
db = dz

w_new = w - lr * dw
b_new = b - lr * db

print('a (output):', round(a, 4), 'loss:', round(loss,4))
print('dw:', np.round(dw, 4), 'db:', round(db, 4))
print('Updated w', np.round(w_new, 4), 'Updated b:', round(b_new,4))
