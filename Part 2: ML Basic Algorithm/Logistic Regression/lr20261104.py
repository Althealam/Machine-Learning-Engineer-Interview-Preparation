import math
import numpy as np
# f(x) = 1/(1+exp(-x)), x = wx+b

# 1. MSE Loss -> logistic regression is for classification (not applicable)
# MSE = sum(y_pred-y)**2/n

# dloss/dw = dloss/dy_pred * dy_pred/dw
# dloss/db = dloww/dy_pred * dy_pred/db

# dloss/dy_pred = 2/n*(y_pred-y) = 2/n*error
# dy_pred/dw = f(x)(1-f(x))x
# dy_pred/db = f(x)(1-f(x))

# dloss/dw = 2*error/n f(x)(1-f(x))x
# dloww/db = 2*error/n f(x)(1-f(x))

# 2. BCE loss 
# BCE = -(ylog(y_pred)+(1-y)log(1-y_pred))
# dloss/dy_pred = -(y/y_pred-(1-y)/(1-y_pred)) 
# = -(y(1-y_pred)-(1-y)y_pred)/y_pred(1-y_pred)
# = -(y-y*y_pred-y_pred+y(y_pred))/y_pred(1-y_pred)
# = -(y-y_pred)/y_pred(1-y_pred)

# df(x)/dw = y_pred(1-y_pred)

# dloss/dw = dloss/dy_pred * dy_pred/dw = -(y-y_pred)*x
# dloww/db = dloss/dy_pred * dy_pred/db = -(y-y_pred)


def sigmoid(x):
    return 1/(1+math.exp(-x))


def train_logistic_regression(X, y, epochs = 100, lr=0.01):
    """
    input:
    - X: (n, d) n个样本，每个样本有d个特征
    - y: (n) 每个样本一个标签
    - w: (d) 每个特征一个权重
    - b: () 标量

    training loop:
    1. iterate all epochs
    2. for each epoch, get the xi and yi and yi_hat
    3. get the dw and db
    4. gradient descent
    """
    n, d = X.shape
    w, b = np.zeros(d), 0
    for epoch in range(epochs):
        dw = 0
        db = 0
        for i in range(n):
            xi = X[i]
            yi = y[i]
            y_predi = sigmoid(w@xi+b)
            error = y_predi-yi
            dw+=error*xi
            db+=error
        average_dw = dw/n
        average_db = db/n 
        w-=lr*average_dw
        b-=lr*average_db
    return w, b

X = np.array([[0, 0], [1, 0], [2, 1], [3, 0]])
y = [0, 1, 0, 1]
print(train_logistic_regression(X, y))

            
# 注意事项：
# 1. X要用array，不能写成list的形式，使用np.array
# 2. logistic regression用于分类问题，因此要用BCE，而不是MSE
# 3. 列表用len，数组用shape
# 3. w要初始化为数组，而不是标量 -> 标量可以用w*x，但是数字要用w@x
# 4. dw要初始化为数组