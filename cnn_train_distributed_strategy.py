import numpy as np
import pandas as pd
import os

import tensorflow as tf

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from cnn_model import VggCnnModel

# memory management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=13000)])


# for continued training setting random seed to get same split everytime this script is run
RANDOM_SEED=1836
# data

checkpoint_dir = '/home/alevink/capstone/checkpoints_folder_fit'
checkpoint_file_name = os.path.join(checkpoint_dir, 'ckpt', '{epoch:02d}-{val_mae:.6f}')


folder = '/home/alevink/capstone/nn_images/cnn_images'

data_labels_metric = pd.read_json('/home/alevink/capstone/cnn_data_train.json.gz')[['thumb_name', 'vid_viewcounts']]
data_labels_metric['vid_viewcounts'] = np.log(data_labels_metric['vid_viewcounts'] + 1)
print('data shape', data_labels_metric.shape)




strategy = tf.distribute.MirroredStrategy()



BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
input_shape = (224, 224, 3)


AUTOTUNE = tf.data.AUTOTUNE

def image_prep(image_name, image_size:tuple):
    img = tf.io.read_file(image_name)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img /= 255

    return img


def custom_data_loader(path_to_folder:str, scores:pd.Series, batch_size:int=32, image_size:tuple=(224, 224), 
                        train_test_split:float=0.2, with_distribute:bool=True, perc_per_run:float=.25):
                        
    scores_ = scores.copy()

    scores_['thumb_name'] = scores_['thumb_name'].apply(lambda file: os.path.join(path_to_folder, file))

    shuffle_scores = scores_.sample(frac=1, random_state=RANDOM_SEED)

    train_list = shuffle_scores.iloc[int(train_test_split*shuffle_scores.shape[0]):, :].sample(frac=perc_per_run)
    train_steps = train_list.shape[0] // batch_size

    test_list = shuffle_scores.iloc[:int(train_test_split*shuffle_scores.shape[0]), :].sample(frac=perc_per_run)
    test_steps = test_list.shape[0] // batch_size


    train_ds = tf.data.Dataset.from_tensor_slices((train_list.thumb_name.values, train_list.vid_viewcounts.values))
    test_ds = tf.data.Dataset.from_tensor_slices((test_list.thumb_name.values, test_list.vid_viewcounts.values))

    train_ds = train_ds.map(lambda p, s: (image_prep(p, image_size=image_size), s), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda p, s: (image_prep(p, image_size=image_size), s), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.batch(batch_size).repeat().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.batch(batch_size).repeat().prefetch(buffer_size=AUTOTUNE)

    if with_distribute:
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        test_ds = strategy.experimental_distribute_dataset(test_ds)


    return train_ds, train_steps, test_ds, test_steps
    



with strategy.scope():
    
    # model optim checkpoint
    CnnModel = VggCnnModel(input_shape=input_shape, batch_size=GLOBAL_BATCH_SIZE)

    CnnModel.compile(optimizer=optimizers.Adam(), 
                    loss=losses.MeanSquaredError(reduction=losses.Reduction.NONE), 
                    metrics=[metrics.RootMeanSquaredError(name='rmse'), metrics.MeanAbsoluteError(name='mae')])

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_name, monitor='val_mae', mode='min')
    # check_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)



    # status = checkpoint.restore(check_manager.latest_checkpoint)

    

train_dist_dataset, train_steps, test_dist_dataset, test_steps = custom_data_loader(folder, 
                                                                                   data_labels_metric, 
                                                                                   batch_size=GLOBAL_BATCH_SIZE, 
                                                                                   image_size=input_shape[:2],
                                                                                   perc_per_run=1)


EPOCHS = 100

history = CnnModel.fit(train_dist_dataset, 
                        epochs=EPOCHS,
                        callbacks=[checkpoint,], 
                        validation_data=test_dist_dataset,
                        steps_per_epoch=train_steps,
                        validation_steps=test_steps,
                        verbose=2)




print('Saving Model...')
CnnModel.save('/home/alevink/capstone/saved_model_fit/CNNModel')

results_df = pd.DataFrame(history.history)
results_df.to_json('/home/alevink/capstone/model_history.json')