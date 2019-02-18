from keras.datasets import mnist
from keras import models
from keras import layers
#1、加载 keras中的MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# >>> train_images.shape
# (60000, 28, 28)
# >>> len(train_labels)
# 60000
# >>> train_labels
# array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
#
# >>> test_images.shape
# (10000, 28, 28)
# >>> len(test_labels)
# 10000
# >>> test_labels
# array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)

#2、网络架构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#3、编译步骤
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#4、准备图像数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#5、准备标签
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#6、拟合模型 打印 测试数据的loss 和 acc
network.fit(train_images, train_labels, epochs=5, batch_size=128)
# Epoch 1/5
# 60000/60000 [==============================] - 9s - loss: 0.2524 - acc: 0.9273
# Epoch 2/5
# 51328/60000 [========================>.....] - ETA: 1s - loss: 0.1035 - acc: 0.9692

#7、验证模型在测试集上的性能
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)