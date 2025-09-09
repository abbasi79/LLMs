# https://huggingface.co/datasets/darrow-ai/USClassActions
#https://arxiv.org/pdf/2211.00582
from datasets import load_dataset
from confusion_mat import *
from sklearn.metrics import  confusion_matrix,ConfusionMatrixDisplay



dataset = load_dataset('darrow-ai/USClassActions')


ds = dataset.with_format("numpy")

# word_count=2000
# def select_first_n_words(text, n=word_count):
#     words = text.split()
#     return " ".join(words[:n])


text_all=[]
verdict_all=[]
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
text_all=np.array(text_all)
verdict_all=np.array(verdict_all)
    
 # %% Selecting only winning data as Oyez data is winning network   
# 0 is win at paintiff= petiotner for Oyez data
#1 is loose

indices_win = np.where(verdict_all  == 'win')
text_all= text_all[indices_win]
verdict_all= verdict_all[indices_win]

verdict_all_binary=  np.where(verdict_all == 'win', 0,1) # Replace 'COLLECTION' with 1, others with 0
verdict_all_binary.astype(np.float64)
text_all_Series= pd.Series(text_all)
verdict_all_Series= pd.Series(verdict_all_binary)
text_all_Series=text_vectorization(text_all_Series)


loaded_model = keras.models.load_model('models/model_cnn_lstm_dilation.keras')


y_pred = loaded_model.predict(text_all_Series)
y_pred=np.where(y_pred<=0.5,0,y_pred)
y_pred=np.where(y_pred>0.5,1,y_pred)

# accuracy = np.mean(verdict_all_Series.astype(int) == y_pred[:,0].astype(int) )

# %% confusion matrices

cm = confusion_matrix(verdict_all_Series.astype(int), y_pred[:,0].astype(int))
make_confusion_matrix(cm)




