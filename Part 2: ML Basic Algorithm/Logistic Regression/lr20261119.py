# f(x)=1/(1+exp(-(wx+b)))
# df(x)/dw = f(x)(1-f(x))(-x)

# BCE loss = -(ylog(y_pred)+(1-y)log(1-y_pred))
# dloss/dy_pred = -(y/y_pred-(1-y)/(1-y_pred))
# = -(y(1-y_pred)-(1-y)y_pred)/y_pred(1-y_pred)
# = -(y-y*y_pred-y_pred+y*y_pred)/y_pred(1-y_pred)
# = -(y-y_pred)/y_pred(1-y_pred)
# dy_pred/dx = y_pred(1-y_pred)
# dloss/dx= -(y-y_pred)
# dloss/dw = dloss/dx*(dx/dw) = -(y-y_pred)*x
# dloww/db = dloss/db*(dx/db) = -(y-y_pred)

import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression(X, y, epochs=100, lr=0.01):
    n, d = X.shape
    w, b = np.zeros(d), 0
    for ecpoh in range(epochs):
        dw = 0
        db = 0
        for i in range(n):
            xi = X[i]
            yi = y[i]
            y_predi = sigmoid(w*xi+b)
            error = y_predi-yi
            dw+=error*xi
            db+=error
        average_dw = dw/n
        average_db = db/n
        w-=lr*average_dw
        b-=lr*average_db
    return w, b

X = np.array([0, 1, 0])
y = np.array([0, 1, 0])
print(logistic_regression(X, y))
