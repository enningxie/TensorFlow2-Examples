from __future__ import absolute_import, division, print_function
import tensorflow as tf
import tensorflow_datasets as tfds


# 载入数据
def load_dataset():
    dataset, info = tfds.load('mnist', with_info=True, as_supervised=True)
    mnist_train, mnist_test = dataset['train'], dataset['test']
    # 数据处理
    mnist_train = mnist_train.map(process_data).shuffle(10000).batch(128)
    mnist_test = mnist_test.map(process_data).batch(128)
    return mnist_train, mnist_test


# 归一化
def process_data(features, label):
    features = tf.cast(features, tf.float32)
    features /= 255
    return features, label


# 构建模型
class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__(name='my_model')
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)


# 训练
@tf.function
def train_step(features, label):
    with tf.GradientTape() as tape:
        predictions = my_model(features)
        tmp_loss = loss_function(label, predictions)
    gradients = tape.gradient(tmp_loss, my_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
    train_loss(tmp_loss)
    train_accuracy(label, predictions)


# 测试
@tf.function
def test_step(features, label):
    predictions = my_model(features)
    tmp_loss = loss_function(label, predictions)
    test_loss(tmp_loss)
    test_accuracy(label, predictions)


if __name__ == '__main__':
    # 载入数据
    mnist_train, mnist_test = load_dataset()

    # 准备模型
    my_model = Classifier()
    ## 设置损失函数
    loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
    ## 设置优化器
    optimizer = tf.keras.optimizers.Adam()
    ## 损失/准确率度量
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # 训练/测试模型
    EPOCHS = 5
    for epoch in range(EPOCHS):
        for tmp_features, tmp_label in mnist_train:
            train_step(tmp_features, tmp_label)

        for tmp_features, tmp_label in mnist_test:
            test_step(tmp_features, tmp_label)

        tmplate = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.2f}%, Test loss: {:.4f}, Test accuracy: {:.2f}%'
        print(tmplate.format(epoch + 1,
                             train_loss.result(),
                             train_accuracy.result() * 100,
                             test_loss.result(),
                             test_accuracy.result() * 100))
