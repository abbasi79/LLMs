import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,ConfusionMatrixDisplay
from sklearn.utils import gen_batches
from confusion_mat import *

X_train, X_test, \
y_train, y_test = train_test_split(
    df[['winner_index', 'facts']],
    df['winner_index'],
    test_size=0.2,
    stratify=df['winner_index'],
    random_state=865
)

X_train= shuffled_train['facts']
y_train = shuffled_train['winner_index']
X_test = X_test['facts']


# Before building CNN model, vectorize facts data
text_vectorization = keras.layers.TextVectorization(
    max_tokens=2000, #orig 2000
    output_mode="int",
    output_sequence_length = 500
)

text_vectorization.adapt(X_train)

X_train_processed = text_vectorization(X_train)
X_test_processed = text_vectorization(X_test)



# %% LSTM
def build_model():

    inputs = keras.Input(shape=(500,))
    one_hot = layers.Embedding(input_dim=2000,output_dim=8,input_length=500, mask_zero=True)(inputs)
    x = layers.Bidirectional(layers.LSTM(4))(one_hot) 
    x = layers.Dropout(0.5)(x) 

    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs,outputs)
    model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['accuracy'])
    return model



 # %% CNN + LSTM
from keras import ops


def build_model_CNN_LSTMcheck(): #proposed model

    inputs = keras.Input(shape=(500,))
    x = layers.Embedding(input_dim=2000,output_dim=8,input_length=500, mask_zero=True)(inputs)
    
    x_orig=x
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=7)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x1 = layers.GlobalAveragePooling1D()(x) 



    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=16)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x2 = layers.GlobalAveragePooling1D()(x) 

    
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=30)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x3 = layers.GlobalAveragePooling1D()(x) 
    
    x=x1+x2+x3

    x = layers.Dense(10,activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x=keras.ops.expand_dims(x, 1)

    x = layers.Bidirectional(layers.LSTM(5))(x) # operating in parallel and generate 32 LSTM (embeddings) in the end

    x = layers.Dropout(0.5)(x) # prevent the overfitting

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs,outputs)
    model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['accuracy'])
    return model

# %% CNN LSTM Dilation
def build_model_CNN_LSTMcheck2(): 

    inputs = keras.Input(shape=(500,))
    x = layers.Embedding(input_dim=2000,output_dim=8,input_length=500, mask_zero=True)(inputs)    
    x_orig=x

    x1 = layers.Conv1D(filters=10,kernel_size=3,activation="relu")(x_orig)
    x2 = layers.Conv1D(filters=20,kernel_size=3,activation="relu")(x1)
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu")(x2)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x = layers.GlobalAveragePooling1D()(x) 


    x = layers.Dense(10,activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x=keras.ops.expand_dims(x, 1)

    x = layers.Bidirectional(layers.LSTM(5))(x) # operating in parallel and generate 32 LSTM (embeddings) in the end
    x = layers.Dropout(0.5)(x) # prevent the overfitting
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs,outputs)
    model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['accuracy'])
    return model


# %% Training and Validation, please note we have greyed out f1-score, precision, recall as it require differnt version of keraas. for paper we used the other version(probably older)

k = 4
num_validation_samples = len(X_train) // k
# num_epochs = 25
num_epochs = 60
batch_sizes = 250
all_loss_histories = []
all_val_loss_histories = []  
all_acc_histories = []
all_val_acc_histories = []
all_val_precs_histories=[]
all_val_recall_histories=[]
# keras.utils.set_random_seed(1) 




# For each validation fold, we will train a full set of epochs, and store the history. 
for fold in range(k):
    validation_data = X_train_processed[num_validation_samples * fold:
                           num_validation_samples * (fold + 1)]
    validation_targets = y_train[num_validation_samples * fold:
                           num_validation_samples * (fold + 1)]
    training_data = np.concatenate([
        X_train_processed[:num_validation_samples * fold],
        X_train_processed[num_validation_samples * (fold + 1):]])
    training_targets = np.concatenate([
        y_train[:num_validation_samples * fold],
        y_train[num_validation_samples * (fold + 1):]])
# 
    # model_lstm2 = build_model()#lstm only 
    model_lstm2 =  build_model_CNN_LSTMcheck()#cnn-lstm-dilation
    # model_lstm2 =build_model_CNN_LSTMcheck2()#cnn-lstm
    
    model_lstm2.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # model_lstm2.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', Precision(), Recall()])

    history = model_lstm2.fit(training_data, training_targets, 
                        validation_data = (validation_data,validation_targets), 
                        epochs=num_epochs, batch_size=batch_sizes)
    
# %% used this code for proposed netowrk cnn-lstm-dilation
    y_pred = model_lstm2.predict(validation_data)
    y_pred=np.where(y_pred<=0.5,0,y_pred)
    y_pred=np.where(y_pred>0.5,1,y_pred)
    cm = confusion_matrix(validation_targets, y_pred)
# %%
    
    
    val_loss_history = history.history['val_loss']
    val_acc_history = history.history['val_accuracy']
    loss_history = history.history['loss']
    acc_history = history.history['accuracy']
    # val_precs_history = history.history['precision']
    # val_recall_history = history.history['recall']

    all_val_loss_histories.append(val_loss_history)
    all_loss_histories.append(loss_history)
    all_val_acc_histories.append(val_acc_history)
    all_acc_histories.append(acc_history)
    # all_val_precs_histories.append(val_precs_history)
    # all_val_recall_histories.append(val_recall_history)

average_loss_history = [np.mean([x[i] for x in all_loss_histories]) for i in range(num_epochs)]
average_val_loss_history = [np.mean([x[i] for x in all_val_loss_histories]) for i in range(num_epochs)]
average_acc_history = [np.mean([x[i] for x in all_acc_histories]) for i in range(num_epochs)]
average_val_acc_history = [np.mean([x[i] for x in all_val_acc_histories]) for i in range(num_epochs)]  
# average_val_precs_history = [np.mean([x[i] for x in all_val_precs_histories]) for i in range(num_epochs)]  
# average_val_recall_history = [np.mean([x[i] for x in all_val_recall_histories]) for i in range(num_epochs)]  
# average_val_f1_history=(2*(np.array(average_val_precs_history) *np.array(average_val_recall_history) ))/(np.array(average_val_precs_history)+np.array(average_val_recall_history))
# %% plots validation loss and accuracy

import matplotlib.pyplot as plt 
plt.style.use('ggplot')

plt.plot(average_loss_history,c='r')
plt.plot(average_acc_history,c="r",linestyle="dashed")
plt.plot(average_val_loss_history,c='b')
plt.plot(average_val_acc_history,c='b',linestyle="dashed")
plt.xlabel("Epochs")
plt.legend(['Training Loss','Training Accuracy','Validation Loss','Validation Accuracy'])
plt.show()
# %% Precision, recall and f1 plots
# import matplotlib.pyplot as plt 
# plt.style.use('ggplot')
# plt.plot(average_val_precs_history,c="g")
# plt.plot(average_val_recall_history ,c='b')
# plt.plot(average_val_f1_history,c='r')
# plt.xlabel("Epochs")
# plt.legend(['Precision','Recall', 'F1 - score'])

# %% confusion matrix for cnn-lstm-dilation of validation data

make_confusion_matrix(cm)

# %% save models

import keras

# Assume 'model' is your trained Keras model
model_lstm2.save('models/model_cnn_lstm_dilation.keras') 

# %%% k fold loss and accuracy plots





