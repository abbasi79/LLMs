# https://huggingface.co/datasets/darrow-ai/USClassActions
#https://arxiv.org/pdf/2211.00582
from datasets import load_dataset
from confusion_mat import *
from sklearn.metrics import  confusion_matrix,ConfusionMatrixDisplay

from nltk.stem import PorterStemmer

# Initialize the Porter Stemmer
ps = PorterStemmer()



dataset = load_dataset('darrow-ai/USClassActions')
#https://huggingface.co/datasets/darrow-ai/USClassActionOutcomes_ExpertsAnnotations


ds = dataset.with_format("numpy")

# word_count=2000

# def select_first_n_words(text, n=word_count):
#     words = text.split()
#     return " ".join(words[:n])


text_all=[]
verdict_all=[]
tokens=1000#best 1000
seq_length=900#best900

first_sample=500#best 500
last_sample=2500#best 2500
batch_sized=500
num_epochs=300
lr=0.0005#best0.001---0.005 gave 65% accuracy

def lr_scheduler(epoch):
    if epoch < 300:
        return 0.0005  # Initial learning rate for first 10 epochs
    else:
        return 0.0001  # Reduced learning rate after 10 epochs
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

for ex in ds["train"]:

    text=ex['target_text']
    verdict=ex['verdict']
    text_all.append(text)
    verdict_all.append(verdict)
    
verdict_all=np.array(verdict_all)    
text_all=np.array(text_all)


l=[i for i in range(len(text_all)) if text_all[i] == None]

text_all = [element for index, element in enumerate(text_all) if index not in l] 
verdict_all = [element for index, element in enumerate(verdict_all) if index not in l] 






# %% word count
# text_all_tmp=np.empty(len(text_all), dtype=object)
# for s in range (len(text_all)):
#     ss=select_first_n_words(text_all[s], word_count)
#     text_all_tmp[s]=ss    
# text_all=text_all_tmp
# %%
text_all=np.array(text_all)[first_sample:last_sample]
verdict_all=np.array(verdict_all)[first_sample:last_sample]
verdict_all_binary=verdict_all
verdict_all_binary[verdict_all_binary == 'win'] = 0#orig 0
verdict_all_binary[verdict_all_binary == 'lose'] = 1
verdict_all_binary=verdict_all_binary.astype(np.int64)

text_all_train, text_all_test, verdict_all_binary_train, verdict_all_binary_test = train_test_split(text_all, verdict_all_binary, test_size=0.001, random_state=865)


text_all_train_Series= pd.Series(text_all_train)
verdict_all_train_Series= pd.Series(verdict_all_binary_train).astype(np.int64)

text_all_test_Series= pd.Series(text_all_test)
verdict_all_test_Series= pd.Series(verdict_all_binary_test).astype(np.int64)

# Before building CNN model, vectorize facts data
text_vectorization2 = keras.layers.TextVectorization(
    max_tokens=tokens, #orig 2000
    output_mode="int",
    output_sequence_length = seq_length
)


text_vectorization2.adapt(text_all_train_Series)

# text_all_train_Series = [ps.stem(t) for t in text_all_train_Series]


X_train_processed_new = text_vectorization2(text_all_train_Series)
X_test_processed_new = text_vectorization2(text_all_test_Series)



# %% validation loss new data


def build_model_CNN_LSTMcheck_new(): #best--thisi s the final i used the best

    inputs = keras.Input(shape=(seq_length,))
    x = layers.Embedding(input_dim=tokens,output_dim=8,input_length=seq_length, mask_zero=True)(inputs)
    
    x_orig=x
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=6)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x1 = layers.GlobalAveragePooling1D()(x) 



    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=14)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x2 = layers.GlobalAveragePooling1D()(x) 

    
    x = layers.Conv1D(filters=40,kernel_size=3,activation="relu",dilation_rate=26)(x_orig)
    x = layers.MaxPool1D(pool_size=2,strides=2)(x)
    x3 = layers.GlobalAveragePooling1D()(x) 

       
    x=x1+x2+x3

    x = layers.Dense(10,activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    x=keras.ops.expand_dims(x, 1)
    x = layers.Bidirectional(layers.LSTM(5))(x) # operating in parallel and generate 32 LSTM (embeddings) in the end
    x = layers.Dropout(0.5)(x) # prevent the overfitting
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs,outputs)
    model.compile(optimizer="rmsprop",loss="binary_crossentropy",metrics=['accuracy'])
    return model


# %% validation new data

from keras import ops

X_train_processed_new_train_vald=X_train_processed_new[100:,:]
verdict_all_train_Series_train_vald=np.array(verdict_all_train_Series)[100:]

k = 4
num_validation_samples = len(X_train_processed_new_train_vald) // k
# num_epochs = 25
num_epochs = num_epochs
batch_sizes =batch_sized
all_loss_histories_new = []
all_val_loss_histories_new = []  
all_acc_histories_new = []
all_val_acc_histories_new = []
# keras.utils.set_random_seed(1) 

for fold in range(k):

    
    validation_data_new = X_train_processed_new_train_vald[num_validation_samples * fold:
                           num_validation_samples * (fold + 1)]
    validation_targets_new = verdict_all_train_Series_train_vald[num_validation_samples * fold:
                           num_validation_samples * (fold + 1)]
    training_data_new = np.concatenate([
        X_train_processed_new_train_vald[:num_validation_samples * fold],
       X_train_processed_new_train_vald[num_validation_samples * (fold + 1):]])
    training_targets_new = np.concatenate([
        verdict_all_train_Series_train_vald[:num_validation_samples * fold],
        verdict_all_train_Series_train_vald[num_validation_samples * (fold + 1):]])
# 

    model_lstm2new =  build_model_CNN_LSTMcheck_new()# cnn_lstm_dilatin_ no dropput (used in revision)
    
    model_lstm2new.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])



    historynew = model_lstm2new.fit(training_data_new, training_targets_new, 
                        validation_data = (validation_data_new,validation_targets_new), 
                        epochs=num_epochs, batch_size=batch_sizes)


    # historynew = model_lstm2new.fit(training_data_new, training_targets_new, 
    #                     validation_data = (validation_data_new,validation_targets_new), 
    #                     epochs=num_epochs, batch_size=batch_sizes,callbacks=[lr_callback])    
 
    
    
    val_loss_history_new = historynew.history['val_loss']
    loss_history_new = historynew.history['loss']
    all_val_loss_histories_new.append(val_loss_history_new)
    all_loss_histories_new.append(loss_history_new)

average_loss_history = [np.mean([x[i] for x in all_loss_histories_new]) for i in range(num_epochs)]
average_val_loss_history = [np.mean([x[i] for x in all_val_loss_histories_new]) for i in range(num_epochs)]

# plots

import matplotlib.pyplot as plt 
plt.style.use('ggplot')

plt.plot(average_loss_history,c='r')
plt.plot(average_val_loss_history,c='b')
plt.xlabel("Epochs")
plt.legend(['Training Loss','Validation Loss'])
plt.show()





# %%saving model

# model_lstm2new.save('models/model_lstm2new.keras')
loaded_model = keras.models.load_model('models/model_lstm2new.keras')


# %% confusion matrices of test data


loaded_model = keras.models.load_model('models/model_lstm2new.keras')#newdata_trained 
cases_start=0
cases_end=100



validation_data_new = X_train_processed_new[cases_start:
                       cases_end]
validation_targets_new = verdict_all_train_Series[cases_start:
                       cases_end]


    
y_pred = loaded_model.predict(validation_data_new)
y_pred=np.where(y_pred<=0.5,0,y_pred)
y_pred=np.where(y_pred>0.5,1,y_pred)


cm = confusion_matrix(validation_targets_new, y_pred)
    

make_confusion_matrix(cm)



