import random
import numpy as np
SEEDS = 46946
random.seed(SEEDS)
np.random.seed(SEEDS)
import tensorflow as tf
tf.random.set_seed(SEEDS)
tf.compat.v1.set_random_seed(SEEDS)
tf.compat.v1.disable_v2_behavior()
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import regularizers
from tensorflow.keras import layers as ll
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from tensorflow.keras.initializers import glorot_normal, lecun_normal
from crc_classiser_utils import recall_m, precision_m, f1score_m, f2score_m


# https://viso.ai/deep-learning/resnet-residual-neural-network/
# https://github.com/viig99/mkscancer/tree/e43ec1feab4b03aebfce8ac4dcdc3ac528ac56e0
# https://machinelearningknowledge.ai/keras-implementation-of-resnet-50-architecture-from-scratch/
# 实现变异的ResNet50在1DCNN上
def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''
    corresponds to the case where the input activation has the same dimension as the output activation
    :param filters:
    :param stage:
    :param block:
    :return:
    '''
    weight_decay = 0.01
    filter1, filter2, filter3 = filters
    conv_name_base = "stage" + str(stage) + "_block" + block + "_conv"
    bn_name_base = "stage" + str(stage) + "_block" + block + "_batch"
    X_shortcut = input_tensor
    X = Conv1D(filters=filter1, kernel_size=1, strides=1, name=conv_name_base+"2a", padding='valid',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=lecun_normal(SEEDS))(input_tensor)
    X = BatchNormalization(axis=(-1), name=bn_name_base + '2a')(X)
    X = Activation('selu')(X)
    X = Conv1D(filters=filter2, kernel_size=kernel_size, strides=1, name=conv_name_base + "2b", padding='same',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=lecun_normal(SEEDS))(X)
    X = BatchNormalization(axis=(-1), name=bn_name_base + '2b')(X)
    X = Activation('selu')(X)
    X = Conv1D(filters=filter3, kernel_size=1, strides=1, name=conv_name_base + "2c", padding='valid',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=lecun_normal(SEEDS))(X)
    X = BatchNormalization(axis=(-1), name=bn_name_base + '2c')(X)
    x = ll.Add()([X, X_shortcut])
    x = Activation('selu')(x)
    # x = BatchNormalization(name=bn_name_base + '2d')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    '''
    the input and output dimensions don’t match up
    there is a CONV1D layer in the shortcut path
    :return:
    '''
    filter1, filter2, filter3 = filters
    weight_decay = 0.01
    conv_name_base = "stage" + str(stage) + "_block" + block + "_conv"
    bn_name_base = "stage" + str(stage) + "_block" + block + "_batch"
    X = Conv1D(filters=filter1, kernel_size=1, strides=strides, name=conv_name_base + "2a", padding='valid',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=lecun_normal(SEEDS))(input_tensor)
    X = BatchNormalization(axis=(-1), name=bn_name_base + '2a')(X)
    X = Activation('selu')(X)
    X = Conv1D(filters=filter2, kernel_size=kernel_size, strides=1, name=conv_name_base + "2b", padding='same',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=lecun_normal(SEEDS))(X)
    X = BatchNormalization(axis=(-1), name=bn_name_base + '2b')(X)
    X = Activation('selu')(X)
    X = Conv1D(filters=filter3, kernel_size=1, name=conv_name_base + "2c", padding='valid', strides=1,
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=lecun_normal(SEEDS))(X)
    X = BatchNormalization(axis=(-1), name=bn_name_base + '2c')(X)
    shortcut = Conv1D(filter3, kernel_size=1, strides=strides, name=conv_name_base + '1',
                      kernel_regularizer=regularizers.l2(weight_decay),
                      kernel_initializer=lecun_normal(SEEDS))(input_tensor)
    shortcut = BatchNormalization(axis=(-1), name=bn_name_base + '1')(shortcut)
    X = ll.Add()([X, shortcut])
    X = Activation('selu')(X)
    return X


# https://stackoverflow.com/questions/45799926/why-batch-normalization-over-channels-only-in-cnn
# CONV中的BatchNormalization在哪个轴上进行
# Input will be (batch, steps, features)
def resnet50(n_class, n_features, params):
    X_input = Input((int(n_features/int(params['channel'])), int(params['channel'])))
    weight_decay = 0.01
    x = Conv1D(filters=64, kernel_size=7, strides=2, name='conv1', padding='same',
               kernel_regularizer=regularizers.l2(weight_decay),
               kernel_initializer=lecun_normal(SEEDS))(X_input)
    x = BatchNormalization(axis=(-1), name='bn1')(x)
    x = Activation('selu')(x)
    x = MaxPooling1D(pool_size=1, strides=2)(x)
    x = conv_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', strides=1)
    x = identity_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='b')
    x = identity_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='c')

    x = conv_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a', strides=2)
    x = identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='b')
    x = identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='c')
    x = identity_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='d')

    x = conv_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a', strides=2)
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='b')
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='c')
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='d')
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='e')
    x = identity_block(x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='f')

    x = conv_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a', strides=2)
    x = identity_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='b')
    x = identity_block(x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='c')

    x = AveragePooling1D(pool_size=int(params['avg_pool']), strides=int(params['avg_pool_s']), name="avg_pool")(x)  #
    x = Flatten()(x)
    x = Dense(n_class, activation='softmax', name="fc1", kernel_initializer=lecun_normal(SEEDS))(x)
    lossf = tf.keras.losses.CategoricalCrossentropy()
    model = Model(inputs=X_input, outputs=x, name=params['method'])
    # print(model.summary())
    optim2 = keras.optimizers.Nadam(lr=float(params['lr']), beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    optim = keras.optimizers.Adam(lr=float(params['lr']), beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    optim1 = tf.keras.optimizers.RMSprop(lr=float(params['lr']),
                                         momentum=0.9, epsilon=1e-06, centered=True)
    optim3 = keras.optimizers.SGD(learning_rate=float(params['lr']), momentum=0.9, decay=1e-07)
    f1 = tfa.metrics.FBetaScore(num_classes=6, average='macro', beta=2., threshold=0.5)
    if params['optimozer'] == "nadam":
        model.compile(loss=lossf, optimizer=optim2,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"), recall_m, precision_m, f1score_m,
                               f2score_m,
                               tf.keras.metrics.Recall(class_id=0, top_k=1, name="recall_0"),
                               tf.keras.metrics.Recall(class_id=1, top_k=1, name="recall_1"),
                               tf.keras.metrics.Recall(class_id=2, top_k=1, name="recall_2"),
                               tf.keras.metrics.Recall(class_id=3, top_k=1, name="recall_3"),
                               tf.keras.metrics.Recall(class_id=4, top_k=1, name="recall_4"),
                               tf.keras.metrics.Recall(class_id=5, top_k=1, name="recall_5"),
                               tf.keras.metrics.Precision(class_id=0, top_k=1, name="precision_0"),
                               tf.keras.metrics.Precision(class_id=1, top_k=1, name="precision_1"),
                               tf.keras.metrics.Precision(class_id=2, top_k=1, name="precision_2"),
                               tf.keras.metrics.Precision(class_id=3, top_k=1, name="precision_3"),
                               tf.keras.metrics.Precision(class_id=4, top_k=1, name="precision_4"),
                               tf.keras.metrics.Precision(class_id=5, top_k=1, name="precision_5")])
    elif params['optimozer'] == "adam":
        model.compile(loss=lossf, optimizer=optim,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"), recall_m, precision_m, f1score_m,
                               f2score_m,
                               tf.keras.metrics.Recall(class_id=0, top_k=1, name="recall_0"),
                               tf.keras.metrics.Recall(class_id=1, top_k=1, name="recall_1"),
                               tf.keras.metrics.Recall(class_id=2, top_k=1, name="recall_2"),
                               tf.keras.metrics.Recall(class_id=3, top_k=1, name="recall_3"),
                               tf.keras.metrics.Recall(class_id=4, top_k=1, name="recall_4"),
                               tf.keras.metrics.Recall(class_id=5, top_k=1, name="recall_5"),
                               tf.keras.metrics.Precision(class_id=0, top_k=1, name="precision_0"),
                               tf.keras.metrics.Precision(class_id=1, top_k=1, name="precision_1"),
                               tf.keras.metrics.Precision(class_id=2, top_k=1, name="precision_2"),
                               tf.keras.metrics.Precision(class_id=3, top_k=1, name="precision_3"),
                               tf.keras.metrics.Precision(class_id=4, top_k=1, name="precision_4"),
                               tf.keras.metrics.Precision(class_id=5, top_k=1, name="precision_5")])
    elif params['optimozer'] == "rmsprop":
        model.compile(loss=lossf, optimizer=optim1,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), recall_m, precision_m, f1score_m,
                               f2score_m,
                               tf.keras.metrics.Recall(class_id=0, top_k=1, name="recall_0"),
                               tf.keras.metrics.Recall(class_id=1, top_k=1, name="recall_1"),
                               tf.keras.metrics.Recall(class_id=2, top_k=1, name="recall_2"),
                               tf.keras.metrics.Recall(class_id=3, top_k=1, name="recall_3"),
                               tf.keras.metrics.Recall(class_id=4, top_k=1, name="recall_4"),
                               tf.keras.metrics.Recall(class_id=5, top_k=1, name="recall_5"),
                               tf.keras.metrics.Precision(class_id=0, top_k=1, name="precision_0"),
                               tf.keras.metrics.Precision(class_id=1, top_k=1, name="precision_1"),
                               tf.keras.metrics.Precision(class_id=2, top_k=1, name="precision_2"),
                               tf.keras.metrics.Precision(class_id=3, top_k=1, name="precision_3"),
                               tf.keras.metrics.Precision(class_id=4, top_k=1, name="precision_4"),
                               tf.keras.metrics.Precision(class_id=5, top_k=1, name="precision_5")])
    elif params['optimozer'] == "sgd":
        model.compile(loss=lossf, optimizer=optim3,
                      metrics=[tf.keras.metrics.CategoricalAccuracy(), recall_m, precision_m, f1score_m, f2score_m])
    else:
        print("set your optimozer!")
    return model
