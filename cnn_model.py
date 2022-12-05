import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Input
from tensorflow.keras import Model






class VggCnnModel(Model):
    def __init__(self, input_shape:tuple=(224, 224, 3), batch_size:int=32, conv_kernel_size:tuple=(3, 3), conv_stride:tuple=(1, 1)) -> None:
        super(VggCnnModel, self).__init__()

        self.input_layer = Input(shape=input_shape, batch_size=batch_size)
        self.conv1_1 = Conv2D(filters=64, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')
        self.conv1_2 = Conv2D(filters=64, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')

        self.max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2_1 = Conv2D(filters=128, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')
        self.conv2_2 = Conv2D(filters=128, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')

        self.max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3_1 = Conv2D(filters=256, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')
        self.conv3_2 = Conv2D(filters=256, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')

        self.max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv4_1 = Conv2D(filters=512, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')
        self.conv4_2 = Conv2D(filters=512, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')
        self.conv4_3 = Conv2D(filters=512, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')

        self.max_pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv5_1 = Conv2D(filters=512, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')
        self.conv5_2 = Conv2D(filters=512, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')
        self.conv5_3 = Conv2D(filters=512, kernel_size=conv_kernel_size, strides=conv_stride, activation='relu', padding='same')

        self.max_pool5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten = Flatten()

        self.dense1 = Dense(4096, activation='relu')
        self.dense2 = Dense(2000, activation='relu')
        self.output_layer = Dense(1, activation='linear')


    def call(self, x):
        
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.max_pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max_pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.max_pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.max_pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.max_pool5(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.output_layer(x)

        return x