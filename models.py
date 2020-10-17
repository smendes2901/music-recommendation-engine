from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Conv3D, BatchNormalization, Flatten, Dense, Dropout, Input, Reshape, Permute, LSTM, TimeDistributed, Bidirectional

cnn = Sequential(name="CNN")
cnn.add(Conv2D(filters=64, kernel_size=[7, 7], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (122x122x64)
cnn.add(BatchNormalization())
cnn.add(AveragePooling2D(pool_size=[2, 2], strides=2))  # Dim = (61x61x64)

cnn.add(Conv2D(filters=128, kernel_size=[7, 7], strides=2, kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (28x28x128)
cnn.add(BatchNormalization())
cnn.add(AveragePooling2D(pool_size=[2, 2], strides=2))  # Dim = (14x14x128)

cnn.add(Conv2D(filters=256, kernel_size=[3, 3], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (12x12x256)
cnn.add(BatchNormalization())
cnn.add(AveragePooling2D(pool_size=[2, 2], strides=2))  # Dim = (6x6x256)

cnn.add(Conv2D(filters=512, kernel_size=[3, 3], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (4x4x512)
cnn.add(BatchNormalization())
cnn.add(AveragePooling2D(pool_size=[2, 2], strides=2))  # Dim = (2x2x512)

cnn.add(BatchNormalization())
cnn.add(Flatten())  # Dim = (2048)

cnn.add(BatchNormalization())
cnn.add(Dropout(0.6))

cnn.add(Dense(1024, activation="relu",
              kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (1024)
cnn.add(Dropout(0.5))

cnn.add(Dense(256, activation="relu",
              kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (256)
cnn.add(Dropout(0.25))

cnn.add(Dense(64, activation="relu",
              kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (64)
cnn.add(Dense(32, activation="relu",
              kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (32))

cnn.add(Dense(8, activation="softmax",
              kernel_initializer=initializers.he_normal(seed=1)))


cnn_lstm = Sequential(name="CNNLSTM")
cnn_lstm.add(Conv2D(filters=64, kernel_size=[7, 7], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (122x122x64)
cnn_lstm.add(BatchNormalization())
cnn_lstm.add(AveragePooling2D(pool_size=[2, 2], strides=2))  # Dim = (61x61x64)

cnn_lstm.add(Conv2D(filters=128, kernel_size=[7, 7], strides=2, kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (28x28x128)
cnn_lstm.add(BatchNormalization())
# Dim = (14x14x128)
cnn_lstm.add(AveragePooling2D(pool_size=[2, 2], strides=2))

cnn_lstm.add(Conv2D(filters=256, kernel_size=[3, 3], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (12x12x256)
cnn_lstm.add(BatchNormalization())
cnn_lstm.add(AveragePooling2D(pool_size=[2, 2], strides=2))  # Dim = (6x6x256)

cnn_lstm.add(Conv2D(filters=512, kernel_size=[3, 3], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (4x4x512)
cnn_lstm.add(BatchNormalization())
cnn_lstm.add(AveragePooling2D(pool_size=[2, 2], strides=2))  # Dim = (2x2x512)

cnn_lstm.add(BatchNormalization())
cnn_lstm.add(Dropout(0.6))

cnn_lstm.add(Reshape((512, -1)))
cnn_lstm.add(Permute((2, 1)))

cnn_lstm.add(LSTM(128, return_sequences=True, input_shape=(128, 128, 1)))
cnn_lstm.add(LSTM(128, input_shape=(128, 128, 1)))

cnn_lstm.add(Dense(1024, activation="relu",
                   kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (1024)
cnn_lstm.add(Dropout(0.5))

cnn_lstm.add(Dense(256, activation="relu",
                   kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (256)
cnn_lstm.add(Dropout(0.25))

cnn_lstm.add(Dense(64, activation="relu",
                   kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (64)
cnn_lstm.add(Dense(32, activation="relu",
                   kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (32)
cnn_lstm.add(Dense(8, activation="softmax",
                   kernel_initializer=initializers.he_normal(seed=1)))


cnn_bi_lstm = Sequential(name="CNNBiLSTM")
cnn_bi_lstm.add(Conv2D(filters=64, kernel_size=[7, 7], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (122x122x64)
cnn_bi_lstm.add(BatchNormalization())
cnn_bi_lstm.add(AveragePooling2D(
    pool_size=[2, 2], strides=2))  # Dim = (61x61x64)

cnn_bi_lstm.add(Conv2D(filters=128, kernel_size=[7, 7], strides=2, kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (28x28x128)
cnn_bi_lstm.add(BatchNormalization())
cnn_bi_lstm.add(AveragePooling2D(
    pool_size=[2, 2], strides=2))  # Dim = (14x14x128)

cnn_bi_lstm.add(Conv2D(filters=256, kernel_size=[3, 3], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (12x12x256)
cnn_bi_lstm.add(BatchNormalization())
cnn_bi_lstm.add(AveragePooling2D(
    pool_size=[2, 2], strides=2))  # Dim = (6x6x256)

cnn_bi_lstm.add(Conv2D(filters=512, kernel_size=[3, 3], kernel_initializer=initializers.he_normal(
    seed=1), activation="relu"))  # Dim = (4x4x512)
cnn_bi_lstm.add(BatchNormalization())
cnn_bi_lstm.add(AveragePooling2D(
    pool_size=[2, 2], strides=2))  # Dim = (2x2x512)

cnn_bi_lstm.add(BatchNormalization())
cnn_bi_lstm.add(Dropout(0.6))

cnn_bi_lstm.add(Reshape((512, -1)))
cnn_bi_lstm.add(Permute((2, 1)))
cnn_bi_lstm.add(Bidirectional(
    LSTM(128, return_sequences=True, input_shape=(128, 128, 1))))
cnn_bi_lstm.add(LSTM(128, input_shape=(128, 128, 1)))

cnn_bi_lstm.add(Dense(1024, activation="relu",
                      kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (1024)
cnn_bi_lstm.add(Dropout(0.5))

cnn_bi_lstm.add(Dense(256, activation="relu",
                      kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (256)
cnn_bi_lstm.add(Dropout(0.25))

cnn_bi_lstm.add(Dense(64, activation="relu",
                      kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (64)
cnn_bi_lstm.add(Dense(32, activation="relu",
                      kernel_initializer=initializers.he_normal(seed=1)))  # Dim = (32)
cnn_bi_lstm.add(Dense(8, activation="softmax",
                      kernel_initializer=initializers.he_normal(seed=1)))
