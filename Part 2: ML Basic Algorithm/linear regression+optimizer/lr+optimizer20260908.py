# y_pred = wx+b
# loss = (y_pred-y)**2
# dloss/dw = 2*(y_pred-y)*x
# dloss/db = 2*(y_pred-y)

# training data
X = [1, 2, 3, 4, 5]
Y = [3, 5, 7, 9, 11]

# parameter
w = 0
b = 0

# hyperparameter
learning_rate = 0.01
epochs = 10
n = len(X)

# training epoch
for epoch in range(epochs):
    print("=============")
    print("epoch: ", epoch)
    total_loss = 0
    # gradient
    dw = 0
    db = 0

    # iterate all sample in the data
    for i in range(n):
        x = X[i]
        y = Y[i]

        # pred
        y_pred = w*x+b

        # error
        error = y_pred-y

        # add error
        total_loss += error**2

        # compute gradients
        dw+=2*error*x
        db+=2*error

    # average loss
    loss = total_loss/n

    # average gradient
    dw = dw/n
    db = db/n

    # update parameter
    w = w-learning_rate*dw
    b = b-learning_rate*db

    print("w=", w)
    print("b=", b)
    print("loss=", loss)
