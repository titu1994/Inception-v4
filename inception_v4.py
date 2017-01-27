from keras.layers import Input, merge, Dropout, Dense, Flatten, Activation
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras import backend as K

"""
Implementation of Inception Network v4 [Inception Network v4 Paper](http://arxiv.org/pdf/1602.07261v1.pdf) in Keras.
"""

def inception_stem(input): # Input (299,299,3)
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    c = Convolution2D(32, 3, 3, activation='relu', subsample=(2,2), bias=False)(input)
    c = Convolution2D(32, 3, 3, activation='relu', bias=False)(c)
    c = Convolution2D(64, 3, 3, activation='relu', border_mode='same', bias=False)(c)

    c1 = MaxPooling2D((3,3), strides=(2,2))(c)
    c2 = Convolution2D(96, 3, 3, activation='relu', subsample=(2,2), bias=False)(c)

    m = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    c1 = Convolution2D(64, 1, 1, activation='relu', border_mode='same', bias=False)(m)
    c1 = Convolution2D(96, 3, 3, activation='relu', bias=False)(c1)

    c2 = Convolution2D(64, 1, 1, activation='relu', border_mode='same', bias=False)(m)
    c2 = Convolution2D(64, 7, 1, activation='relu', border_mode='same', bias=False)(c2)
    c2 = Convolution2D(64, 1, 7, activation='relu', border_mode='same', bias=False)(c2)
    c2 = Convolution2D(96, 3, 3, activation='relu', border_mode='valid', bias=False)(c2)

    m2 = merge([c1, c2], mode='concat', concat_axis=channel_axis)

    p1 = MaxPooling2D((3,3), strides=(2,2), )(m2)
    p2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2,2), bias=False)(m2)

    m3 = merge([p1, p2], mode='concat', concat_axis=channel_axis)
    m3 = BatchNormalization(axis=channel_axis)(m3)
    m3 = Activation('relu')(m3)
    return m3


def inception_A(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    a1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    a1 = Convolution2D(96, 1, 1, activation='relu', border_mode='same', bias=False)(a1)

    a2 = Convolution2D(96, 1, 1, activation='relu', border_mode='same', bias=False)(input)

    a3 = Convolution2D(64, 1, 1, activation='relu', border_mode='same', bias=False)(input)
    a3 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', bias=False)(a3)

    a4 = Convolution2D(64, 1, 1, activation='relu', border_mode='same', bias=False)(input)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', bias=False)(a4)
    a4 = Convolution2D(96, 3, 3, activation='relu', border_mode='same', bias=False)(a4)

    m = merge([a1, a2, a3, a4], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m

def inception_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    b1 = AveragePooling2D((3,3), strides=(1,1), border_mode='same')(input)
    b1 = Convolution2D(128, 1, 1, activation='relu', border_mode='same', bias=False)(b1)

    b2 = Convolution2D(384, 1, 1, activation='relu', border_mode='same', bias=False)(input)

    b3 = Convolution2D(192, 1, 1, activation='relu', border_mode='same', bias=False)(input)
    b3 = Convolution2D(224, 1, 7, activation='relu', border_mode='same', bias=False)(b3)
    b3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same', bias=False)(b3)

    b4 = Convolution2D(192, 1, 1, activation='relu', border_mode='same', bias=False)(input)
    b4 = Convolution2D(192, 1, 7, activation='relu', border_mode='same', bias=False)(b4)
    b4 = Convolution2D(224, 7, 1, activation='relu', border_mode='same', bias=False)(b4)
    b4 = Convolution2D(224, 1, 7, activation='relu', border_mode='same', bias=False)(b4)
    b4 = Convolution2D(256, 7, 1, activation='relu', border_mode='same', bias=False)(b4)

    m = merge([b1, b2, b3, b4], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m

def inception_C(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    c1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(input)
    c1 = Convolution2D(256, 1, 1, activation='relu', border_mode='same', bias=False)(c1)

    c2 = Convolution2D(256, 1, 1, activation='relu', border_mode='same', bias=False)(input)

    c3 = Convolution2D(384, 1, 1, activation='relu', border_mode='same', bias=False)(input)
    c3_1 = Convolution2D(256, 1, 3, activation='relu', border_mode='same', bias=False)(c3)
    c3_2 = Convolution2D(256, 3, 1, activation='relu', border_mode='same', bias=False)(c3)

    c4 = Convolution2D(384, 1, 1, activation='relu', border_mode='same', bias=False)(input)
    c4 = Convolution2D(192, 1, 3, activation='relu', border_mode='same', bias=False)(c4)
    c4 = Convolution2D(224, 3, 1, activation='relu', border_mode='same', bias=False)(c4)
    c4_1 = Convolution2D(256, 3, 1, activation='relu', border_mode='same', bias=False)(c4)
    c4_2 = Convolution2D(256, 1, 3, activation='relu', border_mode='same', bias=False)(c4)

    m = merge([c1, c2, c3_1, c3_2, c4_1, c4_2], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m

def reduction_A(input, k=192, l=224, m=256, n=384):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D((3,3), strides=(2,2))(input)

    r2 = Convolution2D(n, 3, 3, activation='relu', subsample=(2,2), bias=False)(input)

    r3 = Convolution2D(k, 1, 1, activation='relu', border_mode='same', bias=False)(input)
    r3 = Convolution2D(l, 3, 3, activation='relu', border_mode='same', bias=False)(r3)
    r3 = Convolution2D(m, 3, 3, activation='relu', subsample=(2,2), bias=False)(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m

def reduction_B(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    r1 = MaxPooling2D((3, 3), strides=(2, 2))(input)

    r2 = Convolution2D(192, 1, 1, activation='relu', bias=False)(input)
    r2 = Convolution2D(192, 3, 3, activation='relu', subsample=(2, 2), bias=False)(r2)

    r3 = Convolution2D(256, 1, 1, activation='relu', border_mode='same', bias=False)(input)
    r3 = Convolution2D(256, 1, 7, activation='relu', border_mode='same', bias=False)(r3)
    r3 = Convolution2D(320, 7, 1, activation='relu', border_mode='same', bias=False)(r3)
    r3 = Convolution2D(320, 3, 3, activation='relu', border_mode='valid', subsample=(2,2), bias=False)(r3)

    m = merge([r1, r2, r3], mode='concat', concat_axis=channel_axis)
    m = BatchNormalization(axis=channel_axis)(m)
    m = Activation('relu')(m)
    return m


def create_inception_v4(nb_classes=1001):
    '''
    Creates a inception v4 network

    :param nb_classes: number of classes.txt
    :return: Keras Model with 1 input and 1 output
    '''

    if K.image_dim_ordering() == 'th':
        init = Input((3, 299, 299))
    else:
        init = Input((299, 299, 3))

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    x = inception_stem(init)

    # 4 x Inception A
    for i in range(4):
        x = inception_A(x)

    # Reduction A
    x = reduction_A(x)

    # 7 x Inception B
    for i in range(7):
        x = inception_B(x)

    # Reduction B
    x = reduction_B(x)

    # 3 x Inception C
    for i in range(3):
        x = inception_C(x)

    # Average Pooling
    x = AveragePooling2D((8,8))(x)

    # Dropout
    x = Dropout(0.8)(x)
    x = Flatten()(x)

    # Output
    out = Dense(output_dim=nb_classes, activation='softmax')(x)

    model = Model(init, out, name='Inception-v4')
    return model


if __name__ == "__main__":
    from keras.utils.visualize_util import plot

    inception_v4 = create_inception_v4()
    #inception_v4.summary()

    #plot(inception_v4, to_file="Inception-v4.png", show_shapes=True)