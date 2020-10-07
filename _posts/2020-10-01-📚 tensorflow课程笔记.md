---
layout: post
title: 📚 tensorflow课程笔记
img: header_post.jpg
tags: [tensorflow, 📚]
---

- [基础](#基础)
- [预处理](#预处理)
- [均方误差函数 MSE](#均方误差函数-mse)
- [全连接层 Dense](#全连接层-dense)
- [升维 降维](#升维-降维)
- [数组合并](#数组合并)
- [矩阵转置 transpose](#矩阵转置-transpose)
- [范数 tf.norm](#范数-tfnorm)
- [最大 最小 平均](#最大-最小-平均)
- [张量排序](#张量排序)

# 基础

```bash
$ pip install tensorflow
$ pip install matplotlib seaborn
$ pip install Pillow opencv-python opencv-contrib-python
$ pip install scikit-learn
# $ pip install numpy pandas # 已包含
$ sudo apt install python3-tk # linux only
```

```py
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense

def run():
    ds = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = ds.load_data()
    print("train_x.shape:", train_x.shape, ", train_y.shape:", train_y.shape)
    print("test_x.shape:", test_x.shape, ", test_y.shape:", test_y.shape)
    # 正規化
    train_x = tf.keras.utils.normalize(train_x)
    test_x = tf.keras.utils.normalize(test_x)
    # one-hot
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)
    print("train_y.shape:", train_y.shape, ", test_y.shape:", test_y.shape)
    model = tf.keras.Sequential()
    # 模型定义（玄学）
    model.add(Flatten(input_shape=[28, 28]))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    # 模型概要
    model.summary()
    # 编译模型
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    # 训练模型参数
    history = model.fit(train_x, train_y, validation_data=(
        test_x, test_y), epochs=1, batch_size=128)
    score = model.evaluate(test_x, test_y, batch_size=128)
    print("训练评价分数:", score)
    result = model.predict(test_x, batch_size=128)
    print("原始值:", np.argmax(test_y[:20], axis=1))
    print("预测值:", np.argmax(result[:20], axis=1))

run()
```

# 预处理

```py
from tensorflow.keras.preprocessing.sequence import pad_sequences
comment1 = [1, 2, 3, 4]
comment2 = [1, 2, 3, 4, 5, 6, 7]
comment3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
x_train = np.array([comment1, comment2, comment3], dtype=object)
x_test = pad_sequences(x_train) # 左补 0，统一数组长度
x_test = pad_sequences(x_train, value=255) # 左补 255，统一数组长度
x_test = pad_sequences(x_train, padding="post") # 右补 0，统一数组长度
x_test = pad_sequences(x_train, maxlen=3) # 切取数组长度，只保留后 3 位
x_test = pad_sequences(x_train, maxlen=3, truncating="post") # 切取数组长度，只保留前 3 位
```

# 均方误差函数 MSE

```py
rows = 1
out = tf.random.uniform([rows, 10])
print("out:", out)
print("预测值:", tf.math.argmax(out, axis=1), "\n")
y = tf.range(rows)
print("y:", y, "\n")
y = tf.one_hot(y, depth=10)
print("y_one_hot:", y, "\n")
loss = tf.keras.losses.mse(y, out)
print("row loss:", loss, "\n")
loss = tf.reduce_mean(loss)
print("总体损失:", loss, "\n")
```

# 全连接层 Dense

```py
# Dense: y = wx + b
rows = 1
net = tf.keras.layers.Dense(1) # 1 个隐藏层，1 个神经元
net.build((rows, 1)) # 每个训练数据有 1 个特征
print("net.w:", net.kernel) # 参数个数
print("net.b:", net.bias) # 和 Dense 数一样
```

# 升维 降维

```py
a = tf.range([24])
a = tf.reshape(a, [4, 6])
print(a.shape)
print(a.ndim)
# 增加一个维度，相当于 [1,2,3]->[[1,2,3]]
b = tf.expand_dims(a, axis=0)
print(b.shape)
print(b.ndim)
# 减少维度，相当于 [[1,2,3]]->[1,2,3]
c = tf.squeeze(b, axis=0)
print(c.shape)
print(c.ndim)
```

# 数组合并

```py
a = tf.zeros([2, 4, 3])
b = tf.ones([2, 4, 3])
# tf.concat 0 轴合并，4,4,3
c = tf.concat([a, b], axis=0)
# tf.concat 1 轴合并，2,8,3
c = tf.concat([a, b], axis=1)
# tf.concat 2 轴合并，2,4,6
c = tf.concat([a, b], axis=2)
# tf.stack 扩充一维，例如把多个图片放入一个大数组中 -> 2,2,4,3
c = tf.stack([a, b], axis=0)
# tf.unstack 降低维数，拆分数组
m, n = tf.unstack(c, axis=0)
# tf.broadcast_to 数组广播
a = tf.constant([1, 2, 3])
x = 1
print(a + x)
b = tf.broadcast_to(a, [3, 3])
x = 10
print(b * x)
```

# 矩阵转置 transpose

```py
a = tf.range([12])
a = tf.reshape(a, [4, 3])
print(a)
b = tf.transpose(a) # 行列交换
print(b)
# 1 张 4x4 像素的彩色图片
a = tf.random.uniform([4, 4, 3], minval=0, maxval=10, dtype=tf.int32)
print(a)
# 指定变换的轴索引
b = tf.transpose(a, perm=[0, 2, 1])
print(b)
# 把刚才的 b 再变换回来
c = tf.transpose(b, perm=[0, 2, 1])
print(c)
```

# 范数 tf.norm

```py
# 2 范数：平方和开根号
a = tf.fill([1, 2], value=2.)
log("a:", a)
b = tf.norm(a) # 计算 a 的范数
log("a的2范数b:", b)
# 计算验证
a = tf.square(a)
log("a的平方:", a)
a = tf.reduce_sum(a)
log("a平方后的和:", a)
b = tf.sqrt(a)
log("a平方和后开根号:", b)

# 1 范数：所有值的绝对值之和
a = tf.range(12, dtype=tf.float32)
a = tf.reshape(a, (4, 3))
a = a - 5
log("a:", a)
b = tf.norm(a, ord=1)
log("a的1范数b:", b)
print(tf.reduce_sum(tf.abs(a)))
b = tf.norm(a, ord=1, axis=0)
log("a的axis=0的1范数b:", b)
```

# 最大 最小 平均

```py
a = tf.range(12, dtype=tf.float32)
a = tf.reshape(a, (4, 3))
log("a数组:", a)

b = tf.reduce_min(a)
log("a数组最小值:", b)
b = tf.reduce_max(a)
log("a数组最大值:", b)
b = tf.reduce_mean(a)
log("a数组平均值:", b)

b = tf.reduce_min(a, axis=0)
log("a数组axis=0最小值:", b)
b = tf.reduce_max(a, axis=0)
log("a数组axis=0最大值:", b)
b = tf.reduce_mean(a, axis=0)
log("a数组axis=0平均值:", b)

b = tf.argmax(a, axis=1)
log("a数组axis=1的最大值索引位置:", b)
b = tf.argmin(a, axis=1)
log("a数组axis=1的最小值索引位置:", b)
```

# 张量排序

```py
# 1维张量排序
a = tf.random.shuffle(tf.range(10))
log("a", a)
# a tf.Tensor([8 2 0 5 7 9 3 1 4 6], shape=(10,), dtype=int32)
# 升序排列
b = tf.sort(a, direction="ASCENDING")
log("b", b)
# 降序排列
b = tf.sort(a, direction="DESCENDING")
log("b", b)
# b tf.Tensor([0 1 2 3 4 5 6 7 8 9], shape=(10,), dtype=int32)
# b tf.Tensor([9 8 7 6 5 4 3 2 1 0], shape=(10,), dtype=int32)
# 升序排列，返回索引位置
b = tf.argsort(a, direction="ASCENDING")
log("b", b)
# 降序排列，返回索引位置
b = tf.argsort(a, direction="DESCENDING")
log("b", b)
# b tf.Tensor([2 7 1 6 8 3 9 4 0 5], shape=(10,), dtype=int32)
# b tf.Tensor([5 0 4 9 3 8 6 1 7 2], shape=(10,), dtype=int32)
# 按索引位置b, 从数组a中收集数据
c = tf.gather(a, b)
log("c", c)
# c tf.Tensor([9 8 7 6 5 4 3 2 1 0], shape=(10,), dtype=int32)

# 2维张量排序
a = tf.random.uniform([3, 5], maxval=10, dtype=tf.int32)
log("a", a)
# a tf.Tensor([[2 5 8 0 4] [1 7 2 4 5] [6 0 2 5 0]], shape=(3, 5), dtype=int32)
# 升序排列
b = tf.sort(a, axis=1, direction="ASCENDING")
log("b", b)
# 降序排列
b = tf.sort(a, axis=1, direction="DESCENDING")
log("b", b)
# b tf.Tensor([[0 2 4 5 8] [1 2 4 5 7] [0 0 2 5 6]], shape=(3, 5), dtype=int32)
# b tf.Tensor([[8 5 4 2 0] [7 5 4 2 1] [6 5 2 0 0]], shape=(3, 5), dtype=int32)
# 升序排列，返回索引位置
b = tf.argsort(a, axis=1, direction="ASCENDING")
log("b", b)
# 降序排列，返回索引位置
b = tf.argsort(a, axis=1, direction="DESCENDING")
log("b", b)
# b tf.Tensor([[3 0 4 1 2] [0 2 3 4 1] [1 4 2 3 0]], shape=(3, 5), dtype=int32)
# b tf.Tensor([[2 1 4 0 3] [1 4 3 2 0] [0 3 2 1 4]], shape=(3, 5), dtype=int32)

# top 值
a = tf.random.uniform([3, 5], maxval=10, dtype=tf.int32)
log("a", a)
# a tf.Tensor([[1 2 5 8 7] [6 1 4 3 9] [5 9 6 5 5]], shape=(3, 5), dtype=int32)
# 取数组每行前3位
b = tf.math.top_k(a, k=3, sorted=True)
log("b", b.values) # 前3位数值
log("b", b.indices) # 前3位数值索引
# b tf.Tensor([[8 7 5] [9 6 4] [9 6 5]], shape=(3, 3), dtype=int32)
# b tf.Tensor([[3 4 2] [4 0 2] [1 2 0]], shape=(3, 3), dtype=int32)
```
