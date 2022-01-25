import tensorflow as tf
from tensorflow.keras import layers, Sequential, optimizers


def block(layer, filters, kernel_size, strides=1, padding='valid', layer_name='conv',
          pool_size=2, pool_strides=None, filter_2=False):
    if layer_name == 'conv':
        layer = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                       strides=strides, use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        layer = tf.keras.layers.Conv2D(filters=filters * 2, kernel_size=kernel_size,
                                       use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)

    elif layer_name == 'separable_conv':
        layer = tf.keras.layers.SeparableConv2D(filters, kernel_size,
                                                padding=padding, use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.ReLU()(layer)
        if filter_2:
            layer = tf.keras.layers.SeparableConv2D(filter_2, kernel_size,
                                                    padding=padding, use_bias=False)(layer)
        else:
            layer = tf.keras.layers.SeparableConv2D(filters, kernel_size,
                                                    padding=padding, use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.MaxPooling2D(pool_size, strides=pool_strides,
                                             padding=padding)(layer)
    return layer


def add_block(layer, filters, kernel_size, strides=1, padding='valid', pool_size=2, pool_strides=None):
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.SeparableConv2D(filters, kernel_size, padding=padding, use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.MaxPooling2D(pool_size, strides=pool_strides,
                                         padding=padding)(layer)
    return layer

def entry_flow(input_layer):
  block_1 = block(input_layer,32,3,2,layer_name='conv')

  block_2 = block(block_1,128,3,padding='same',layer_name='separable_conv')
  layer_add = tf.keras.layers.Conv2D(filters=128,kernel_size=1,strides=2,
                                 padding='same',use_bias=False)(block_1)
  layer_add = tf.keras.layers.BatchNormalization()(layer_add)
  layer = tf.keras.layers.Add()([block_2,layer_add])

  block_3 = add_block(layer,256,3,1,'same',3,2)
  layer_add = tf.keras.layers.Conv2D(filters=256,kernel_size=1,strides=2,
                                 padding='same',use_bias=False)(layer)
  layer_add = tf.keras.layers.BatchNormalization()(layer_add)
  layer = tf.keras.layers.Add()([block_3,layer_add])

  block_4 = add_block(layer,728,3,1,'same',3,2)
  layer_add = tf.keras.layers.Conv2D(filters=728,kernel_size=1,strides=2,
                                 padding='same',use_bias=False)(layer)
  layer_add = tf.keras.layers.BatchNormalization()(layer_add)
  layer = tf.keras.layers.Add()([block_4,layer_add])
  return layer

def middle_flow(input_layer):
    for _ in range(8):
      for __ in range(3):
        layer = tf.keras.layers.ReLU()(input_layer)
        layer = tf.keras.layers.SeparableConv2D(filters=728, kernel_size=3,
                                                padding='same',use_bias=False)(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
      output_layer = tf.keras.layers.Add()([input_layer, layer])
    return output_layer


def exit_flow(input_layer):
    layer = tf.keras.layers.ReLU()(input_layer)
    block_1 = block(layer, 728, 3, padding='same', layer_name='separable_conv',
                    pool_size=3, pool_strides=2, filter_2=1024)

    layer_add = tf.keras.layers.Conv2D(filters=1024, kernel_size=1,
                                       strides=2, padding='same', use_bias=False)(input_layer)
    layer_add = tf.keras.layers.BatchNormalization()(layer_add)
    layer = tf.keras.layers.Add()([block_1, layer_add])

    layer = tf.keras.layers.SeparableConv2D(filters=1536, kernel_size=3,
                                            padding='same', use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)
    layer = tf.keras.layers.SeparableConv2D(filters=2048, kernel_size=3,
                                            padding='same', use_bias=False)(layer)
    layer = tf.keras.layers.BatchNormalization()(layer)
    layer = tf.keras.layers.ReLU()(layer)

    layer = tf.keras.layers.GlobalAvgPool2D()(layer)
    layer = tf.keras.layers.Dense(64, activation='relu')(layer)
    layer = tf.keras.layers.Dropout(0.3)(layer)

    return layer

def xception(shape,include_top):
    model_input = tf.keras.layers.Input(shape=shape)
    entry_block = entry_flow(model_input)
    mid_block = middle_flow(entry_block)
    exit_block = exit_flow(mid_block)

    if include_top:
        model_output = tf.keras.layers.Dense(1, activation='sigmoid')(exit_block)
        model = tf.keras.models.Model(model_input, model_output)
    model = tf.keras.models.Model(model_input, model_output)
    model.summary()
    return model

def train_model(train_x, train_y, test_x, test_y, epoch, btch, width):
    shape = width, width, 3
    model = xception(shape, include_top=True)
    starter_learning_rate = 1e-2
    end_learning_rate = 1e-12
    decay_steps = 1000
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        starter_learning_rate,
        decay_steps,
        end_learning_rate,
        power=0.8)
    optimizer = optimizers.Adam(learning_rate = learning_rate_fn) #学习率

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='Liverre_{epoch}_{val_accuracy:.2f}',
            save_best_only=True,
            monitor='val_accuracy',
            save_weights_only=True,
            verbose=1)]
    # total = train_y.shape[0]
    # labels_dict = dict(zip(range(num_task), [sum(train_y[:, i]) for i in range(num_task)]))
    # cls_wght = create_class_weight(labels_dict, total)

    history = model.fit(train_x, train_y, epochs=epoch, batch_size=btch, validation_data=(test_x , test_y), callbacks=callbacks)

    print ('testing the model')
    score = model.evaluate(train_x, train_y)
    print("train_score: ", score)
    score = model.evaluate(test_x , test_y )
    print("test_score: " , score)