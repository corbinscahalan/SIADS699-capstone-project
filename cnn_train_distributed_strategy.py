import pandas as pd
import os

import tensorflow as tf

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from cnn_model import VggCnnModel


# for continued training. setting random seed to get same split everytime 
# keeping test and train separate across training sessions
RANDOM_SEED=1836

# checkpoint directory
checkpoint_dir = '/checkpoints_folder'

# image folder
folder = '/nn_images/cnn_images'

# files and labels
data_labels_metric = pd.read_json('/cnn_data_train.json.gz')[['thumb_name', 'pop_metric']]
print('data shape', data_labels_metric.shape)



# strategy for distibuting on multiple gpus
strategy = tf.distribute.MirroredStrategy()


# batch and input size
BATCH_SIZE_PER_REPLICA = 64
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
input_shape = (224, 224, 3)


AUTOTUNE = tf.data.AUTOTUNE

# read in image, resize and normalize
def image_prep(image_name, image_size:tuple):
    img = tf.io.read_file(image_name)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img /= 255

    return img

# loadf the data and map files to score, in a tf.data.Dataset
def custom_data_loader(path_to_folder:str, scores:pd.Series, batch_size:int=32, image_size:tuple=(224, 224), 
                        train_test_split:float=0.2, with_distribute:bool=True, perc_per_run:float=1):
                        
    scores_ = scores.copy()

    # transform file names to file paths
    scores_['thumb_name'] = scores_['thumb_name'].apply(lambda file: os.path.join(path_to_folder, file))

    # shuffle but use random seed to keep the same for each run
    shuffle_scores = scores_.sample(frac=1, random_state=RANDOM_SEED)

    train_list = shuffle_scores.iloc[int(train_test_split*shuffle_scores.shape[0]):, :].sample(frac=perc_per_run)

    test_list = shuffle_scores.iloc[:int(train_test_split*shuffle_scores.shape[0]), :].sample(frac=perc_per_run)

    # transform into a Dataset
    train_ds = tf.data.Dataset.from_tensor_slices((train_list['thumb_name'].values, train_list['pop_metric'].values))
    test_ds = tf.data.Dataset.from_tensor_slices((test_list['thumb_name'].values, test_list['pop_metric'].values))

    train_ds = train_ds.map(lambda p, s: (image_prep(p, image_size=image_size), s), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda p, s: (image_prep(p, image_size=image_size), s), num_parallel_calls=AUTOTUNE)
    # .cache().shuffle(buffer_size=1000) took up too much memory
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    # if using a distributed strategy
    if with_distribute:
        train_ds = strategy.experimental_distribute_dataset(train_ds)
        test_ds = strategy.experimental_distribute_dataset(test_ds)


    return train_ds, test_ds



# transformed loaded Dataset
train_dist_dataset, test_dist_dataset = custom_data_loader(folder, 
                                                            data_labels_metric, 
                                                            batch_size=GLOBAL_BATCH_SIZE, 
                                                            image_size=input_shape[:2],
                                                            perc_per_run=1)



# items that must be declare under the strategy scope
with strategy.scope():
    # train loss
    mse_loss = losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
      
    def compute_loss(labels, predictions):
        #print(labels.shape, predictions.shape)
        per_example_loss = mse_loss(tf.reshape(labels, [labels.shape[0], 1]), predictions)
        #print(per_example_loss.shape)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    
    # metrics
    train_metric = metrics.RootMeanSquaredError(name='train_rmse')
    test_loss = metrics.Mean(name='test_loss')
    test_metric = metrics.RootMeanSquaredError(name='train_rmse')
    
    # model optim checkpoint
    CnnModel = VggCnnModel(input_shape=input_shape, batch_size=GLOBAL_BATCH_SIZE)

    adam_optim = optimizers.Adam()

    checkpoint = tf.train.Checkpoint(optimizer=adam_optim, model=CnnModel)
    check_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)



    status = checkpoint.restore(check_manager.latest_checkpoint)

    def train_step(data_inputs):
        images, values = data_inputs
        with tf.GradientTape() as tape:

            preds = CnnModel(images)
            tr_loss = compute_loss(values, preds)

            gradients = tape.gradient(tr_loss, CnnModel.trainable_variables)
            adam_optim.apply_gradients(zip(gradients, CnnModel.trainable_variables))

            train_metric.update_state(values, preds)
            return tr_loss

    def test_step(data_inputs):
        images, values = data_inputs

        preds = CnnModel(images)
        te_loss = mse_loss(values, preds)

        test_loss.update_state(te_loss)
        test_metric.update_state(values, preds)

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))


    


data_rec = pd.DataFrame(columns=['epoch', 'train_loss', 'train_rmse', 'test_loss', 'test_rmse']).set_index('epoch', drop=True)




EPOCHS = 200

for epoch in range(EPOCHS):
    
    train_metric.reset_states()
    test_loss.reset_states()
    test_metric.reset_states()

    total_loss = 0.0
    train_batch = 0

    for data in train_dist_dataset:
        train_batch += 1
        if train_batch % 50 == 0:
            print(f'----- Training EPOCH: {epoch}, batch: {train_batch}, samples seen: {train_batch*GLOBAL_BATCH_SIZE}')
        total_loss += distributed_train_step(data)
    status.assert_consumed()
    train_loss = total_loss / train_batch

    test_batch = 0
    for data in test_dist_dataset:
        test_batch += 1
        if test_batch % 50 == 0:
            print(f'----- Testing EPOCH: {epoch}, batch: {test_batch}, samples seen: {train_batch*GLOBAL_BATCH_SIZE}')
        distributed_test_step(data)

    if epoch % 2 == 0:
        check_manager.save()
    if epoch % 20 == 0:
        CnnModel.save(f'/saved_model/CNNModel{epoch}')
        

    data_rec.loc[(epoch+1), :] = [str(train_loss), str(train_metric.result()), str(test_loss.result()), str(test_metric.result())]
    
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss}, '
        f'RMSE: {train_metric.result()}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test RMSE: {test_metric.result()}'
    )
data_rec.to_json(f'/model_log/log_dataframe_after_epoch{epoch + 1}.json')
print('Saving Model...')
CnnModel.save('/saved_model/CNNModel')
