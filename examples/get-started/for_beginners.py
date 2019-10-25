from __future__ import absolute_import, division, print_function
import tensorflow as tf
import matplotlib.pyplot as plt


# 载入mnist数据集
def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 归一化
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


# 构建模型
def construct_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


if __name__ == '__main__':
    # 获取数据
    x_train, y_train, x_test, y_test = load_data()

    # 获取模型
    model = construct_model()
    model.summary()
    tf.keras.utils.plot_model(model, './begineers.png', show_shapes=True)

    # 训练模型
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.sparse_categorical_accuracy])
    history = model.fit(x_train, y_train,
                        batch_size=512,
                        epochs=10,
                        validation_split=0.2,
                        shuffle=True)

    # 评价模型
    loss, acc = model.evaluate(x_test, y_test)
    print("model's loss: {:.4f}, acc: {:.4f}.".format(loss, acc))

    # 绘图
    plt.subplot(121)
    plt.plot(history.history['loss'], 'g', label='train_loss')
    plt.plot(history.history['val_loss'], 'r', label='val_loss')
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.subplot(122)
    plt.plot(history.history['sparse_categorical_accuracy'], 'g', label='train_acc')
    plt.plot(history.history['val_sparse_categorical_accuracy'], 'r', label='val_acc')
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
