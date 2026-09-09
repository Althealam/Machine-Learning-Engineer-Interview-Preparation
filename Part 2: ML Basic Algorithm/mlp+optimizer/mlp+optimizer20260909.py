# # 假设输入是一个标量x，隐藏层有两个神经元，输出是一个标量，激活函数是ReLU，loss用MSE
# # 模型结构：x->z->a->y_hat->L

# # forward
# # 1. zi = xwi+bi (i=1, 2)
# # 2. ai = max(0, zi)
# # 3. y_hat = a1v1+a2v2+b

# # backward
# # loss = (y_hat-y)**2
# # dy_hat = 2(y_hat-y)

# # dv1 = dy_hat*a1, dv2 = dy_hat*a2
# # db = dy_hat
# # da1 = dy_hat*v1, da2 = dy_hat*v2

# # dz1 = da1*1 if z1>0 else 0, dz2 = da2*1 if z2>0 else 0
# # dw1 = dz1*x, dw2=dz2*x
# # db1 = dz1, db2 = dz2


# import numpy as np

# # 隐藏层
# w1 = 1
# w2 = 2
# b1 = 0
# b2 = 0

# # 输出层
# v1 = 2
# v2 = 4
# b = 0

# lr = 0.01

# def forward(x, w1, w2, b1, b2, v1, v2, b):
#     # 第一层
#     z1 = x*w1+b1
#     z2 = x*w2+b2

#     # ReLU
#     a1 = max(0, z1)
#     a2 = max(0, z2)

#     # 输出层
#     y_hat = a1*v1+a2*v2+b
#     return z1, z2, a1, a2, y_hat

# def loss_fn(y_hat, y):
#     return (y_hat-y)**2

# def backward(x, y, z1, z2, a1, a2, y_hat, v1, v2):
#     # loss->y_hat
#     dy_hat = 2*(y_hat-y)

#     # output layer
#     dv1 = dy_hat*a1
#     dv2 = dy_hat*a2
#     db = dy_hat

#     # hidden state
#     da1 = dy_hat*v1
#     da2 = dy_hat*v2

#     # relu backward
#     dz1 = da1 if z1>0 else 0
#     dz2 = da2 if z2>0 else 0

#     # first layer
#     dw1 = dz1*x
#     dw2 = dz2*x
#     db1 = dz1
#     db2 = dz2

#     return dw1, dw2, db1, db2, dv1, dv2, db

# # sgd update
# def update(w1, w2, b1, b2, v1, v2, b, dw1, dw2, db1, db2, dv1, dv2, db, lr):
#     w1 = w1-lr*dw1
#     w2 = w2-lr*dw2

#     b1 = b1-lr*db1
#     b2 = b2-lr*db2

#     v1 = v1-lr*dv1
#     v2 = v2-lr*dv2

#     b = b-lr*db

#     return w1, w2, b1, b2, v1, v2, b

# # training 
# x = 2
# y = 5
# for epoch in range(10):
#     # forward
#     z1, z2, a1, a2, y_hat = forward(x, w1, w2, b1, b2, v1, v2, b)
#     loss = loss_fn(y_hat, y)
#     # backward
#     dw1, dw2, db1, db2, dv1, dv2, db = backward(x, y, z1, z2, a1, a2, y_hat, v1, v2)
#     # update 
#     w1, w2, b1, b2, v1, v2, b = update(w1, w2, b1, b2, v1, v2, b, dw1, dw2, db1, db2, dv1, dv2, db, lr)
#     print("epoch:", epoch)
#     print("y_hat:", y_hat)
#     print("loss:", loss)


w1, w2 = 0.5, -0.3
b1, b2 = 0.0, 0.0

v1, v2 = 0.8, -0.2
b = 0.0

lr = 0.01

x = 2
y = 5

for epoch in range(100):
    # forward
    z1 = x*w1+b1
    z2 = x*w2+b2

    a1 = max(0, z1)
    a2 = max(0, z2)

    y_hat = a1*v1+a2*v2+b

    loss = (y_hat-y)**2

    # backward
    dy_hat = 2*(y_hat-y)

    dv1 = dy_hat*a1
    dv2 = dy_hat*a2
    db = dy_hat
    
    da1 = dy_hat*v1
    da2 = dy_hat*v2

    dz1 = da1 if z1>0 else 0
    dz2 = da2 if z2>0 else 0

    dw1 = dz1*x
    dw2 = dz2*x

    db1 = dz1
    db2 = dz2

    # sgd
    w1-=lr*dw1
    w2-=lr*dw2

    b1-=lr*db1
    b2-=lr*db2

    v1-=lr*dv1
    v2-=lr*dv2
    b-=lr*db

    print(epoch, y_hat, loss)