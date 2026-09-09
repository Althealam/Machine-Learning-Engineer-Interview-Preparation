X = [1, 2, 3, 4, 5]
Y = [4, 7, 10, 13, 16]

# 训练过程
# 1. 遍历epoch
# 2. 对每个epoch，遍历所有的训练样本
# 3. 计算y_pred = w*x+b
# 4. 计算error = y_pred-y
# 5. 累加loss
# 6. 计算并累加dw和db
# 7. 对loss, dw, db求平均
# 8. 使用梯度下降更新w和b
# 9. 对每个epoch打印w, b, loss

w, b = 2, 3
epochs = 100
n = len(X)
lr = 0.01
for epoch in range(epochs):
    total_loss = 0
    dw = 0
    db = 0
    for i in range(n):
        x, y = X[i], Y[i]

        # prediction
        y_pred = w*x+b

        # error
        error = y_pred-y

        # loss
        loss = (y_pred-y)**2
        total_loss+=loss

        # gradients
        dw = error*x
        db = error

    # average loss
    average_loss = total_loss/n
    # average gradient
    average_dw = dw/n
    average_db = db/n
    # update w and b
    w-=lr*average_dw
    b-=lr*average_db

    print(epoch, w, b, loss)
