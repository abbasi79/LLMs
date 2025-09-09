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



def build_model_CNN_LSTMcheck2(): #proposed model

    inputs = keras.Input(shape=(500,))
    x = layers.Embedding(input_dim=2000,output_dim=8,input_length=500, mask_zero=True)(inputs)
    
    x_orig=x
    # x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=7)(x_orig)
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=10)(x_orig)

    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x1 = layers.GlobalAveragePooling1D()(x) 



    # x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=16)(x_orig)
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=15)(x_orig)

    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x2 = layers.GlobalAveragePooling1D()(x) 

    
    # x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=30)(x_orig)
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=20)(x_orig)

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



def build_model_CNN_LSTMcheck4(): #proposed model

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
    
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=14)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x4 = layers.GlobalAveragePooling1D()(x) 
    
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=32)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x5 = layers.GlobalAveragePooling1D()(x) 
    
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=60)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x6 = layers.GlobalAveragePooling1D()(x) 
    
    x=x1+x2+x3+x4+x5+x6

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
all_loss_histories2 = []
all_loss_histories3 = []
all_loss_histories4 = []

all_val_loss_histories = []  
all_val_loss_histories2 = []  
all_val_loss_histories3 = []  
all_val_loss_histories4 = []  

all_val_acc_histories=[]
all_val_acc_histories2=[]
all_val_acc_histories3=[]
all_val_acc_histories4=[]


keras.utils.set_random_seed(1) 




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
    model_lstm2 =build_model_CNN_LSTMcheck2()#5-15-20
    model_lstm4 =build_model_CNN_LSTMcheck4()#7-16-30-14-32-60

    

    model_lstm2.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    model_lstm4.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # model_lstm2.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    #               loss='binary_crossentropy',
    #               metrics=['accuracy', Precision(), Recall()])

    history2 = model_lstm2.fit(training_data, training_targets, 
                        validation_data = (validation_data,validation_targets), 
                        epochs=num_epochs, batch_size=batch_sizes)
    

    history4 = model_lstm4.fit(training_data, training_targets, 
                        validation_data = (validation_data,validation_targets), 
                        epochs=num_epochs, batch_size=batch_sizes)
    
# %% used this code for proposed netowrk cnn-lstm-dilation
    # y_pred = model_lstm2.predict(validation_data)
    # y_pred=np.where(y_pred<=0.5,0,y_pred)
    # y_pred=np.where(y_pred>0.5,1,y_pred)
    # cm = confusion_matrix(validation_targets, y_pred)
# %%
    
    
    val_loss_history2 = history2.history['val_loss']
    val_loss_history4 = history4.history['val_loss']

    loss_history2 = history2.history['loss']
    loss_history4 = history4.history['loss']
    
    
    val_acc_history2 = history2.history['val_accuracy']
    val_acc_history4 = history4.history['val_accuracy']

    



    all_val_loss_histories2.append(val_loss_history2)
    all_val_loss_histories4.append(val_loss_history4)

    all_loss_histories2.append(loss_history2)
    all_loss_histories4.append(loss_history4)
    
    
    all_val_acc_histories2.append(val_acc_history2)
    all_val_acc_histories4.append(val_acc_history4)





average_loss_history2 = [np.mean([x[i] for x in all_loss_histories2]) for i in range(num_epochs)]
average_loss_history4 = [np.mean([x[i] for x in all_loss_histories4]) for i in range(num_epochs)]

average_val_loss_history2 = [np.mean([x[i] for x in all_val_loss_histories2]) for i in range(num_epochs)]
average_val_loss_history4 = [np.mean([x[i] for x in all_val_loss_histories4]) for i in range(num_epochs)]



average_val_acc_history2 = [np.mean([x[i] for x in all_val_acc_histories2]) for i in range(num_epochs)]  
average_val_acc_history4 = [np.mean([x[i] for x in all_val_acc_histories4]) for i in range(num_epochs)]  


average_loss_history2_np=np.array(average_loss_history2)
average_loss_history4_np=np.array(average_loss_history4) 
average_val_loss_history2_np=np.array(average_val_loss_history2)
average_val_loss_history4_np=np.array(average_val_loss_history4)    




# %% plots validation loss and accuracy

import matplotlib.pyplot as plt 
plt.style.use('ggplot')

plt.plot(average_loss_history,c='r')
plt.plot(average_val_loss_history,c="m",linestyle="dashed")
plt.plot(average_val_loss_history2,c='b')
plt.plot(average_val_loss_history3,c='g')
plt.plot(average_val_loss_history4,c='k')
plt.xlabel("Epochs")
plt.legend(['Training Loss','Val. Loss 7-16-30 ','Val. Loss 4-10-18','Val. Loss 7-16','Val. Loss 7-16-30-42-64-90'])
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

average_loss_history2_np=np.load('average_loss_history2_np.npy')
average_loss_history4_np=np.load('average_loss_history4_np.npy')
average_val_loss_history4_np=np.load('average_val_loss_history4_np.npy')
average_val_loss_history2_np=np.load('average_val_loss_history2_np.npy')


import matplotlib.pyplot as plt 
plt.style.use('ggplot')





plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(average_loss_history2,c='r')
plt.plot(average_val_loss_history2,c='b')
plt.xlabel("Epochs")
plt.title('(a)')
plt.legend(['Train Loss', 'Val. Loss-Random Dilations : 5-15-20 '], fontsize=8,loc='lower left')
# plt.legend(loc='lower left')


plt.subplot(1, 2, 2)
plt.plot(average_loss_history4,c='r')
plt.plot(average_val_loss_history4,c='b')
plt.xlabel("Epochs")
plt.title('(b)')
plt.legend(['Train Loss','Val. Extra Dilations :  7-16-30-14-32-60'], fontsize=8,loc='lower left')
# plt.legend(loc='lower left')

plt.show()








