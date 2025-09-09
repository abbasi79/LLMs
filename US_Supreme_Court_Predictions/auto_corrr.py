#https://stackoverflow.com/questions/643699/how-can-i-use-numpy-correlate-to-do-autocorrelation
# validation loss less than training loss is ok
#https://stackoverflow.com/questions/67949311/my-validation-loss-is-lower-than-my-training-loss-should-i-get-rid-of-regulariz
import numpy as np
def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]


case_num=100
cc=X_train_processed.numpy()
cc1=cc[case_num,:].flatten('F')


cc1_auto=autocorr(cc1)
plt.plot(cc1_auto[1:])


cc50_auto=autocorr(cc1)
cc100_auto=autocorr(cc1)

plt.plot(cc50_auto)
plt.plot(cc100_auto)

top_3_lags=[]#other than zero
cc=X_train_processed.numpy()


for i in range(cc.shape[0]):
    case_num=i
    cc1=cc[case_num,:].flatten('F')
    cc1_auto=autocorr(cc1)
    top_3_indices = np.sort(np.argpartition(cc1_auto, -4)[-4:])[1:]
    top_3_lags.append(top_3_indices)
    
plt.plot(top_3_lags[::5])
plt.title('Top 3 Non-Zero Lags')
plt.xlabel('Vectorized Cases')
plt.ylabel('Lags')
plt.legend(["Mean Lag 6.5", "Mean Lag 15.9", "Mean Lag 30.4"], loc="upper right")
plt.show()

# mean lags

top_3_lags_down=top_3_lags[::5]
lag1=[]
lag2=[]
lag3=[]

for i in range(len(top_3_lags_down)):
    for j in range(3):
        if j==0:
            lag1.append(top_3_lags[i][j])
        if j==1:
            lag2.append(top_3_lags[i][j])
        if j==2:
            lag3.append(top_3_lags[i][j])
from scipy import stats as st
res = st.mode(lag3)
print(res.mode)

# take eitther mean or mode of each of the lag to estimate dilations      
#lag1=6.5
#lag2=15.9
#lag3=30.4



# %%  Dilation comparison
#best [7,16,30] matches 
#2nd best [2,4,6]
# [2,4,22]
# [2,6,30]
# [2,10,40]

average_val_acc_history_2_4_6=np.array(average_val_acc_history)
average_val_loss_history_2_4_6=np.array(average_val_loss_history)

average_val_acc_history_2_4_22=np.array(average_val_acc_history)
average_val_loss_history_2_4_22=np.array(average_val_loss_history)

average_val_acc_history_2_6_30=np.array(average_val_acc_history)
average_val_loss_history_2_6_30=np.array(average_val_loss_history)

average_val_acc_history_2_10_40=np.array(average_val_acc_history)
average_val_loss_history_2_10_40=np.array(average_val_loss_history)

average_val_acc_history_7_16_30=np.array(average_val_acc_history)
average_val_loss_history_7_16_30=np.array(average_val_loss_history)



# %% accuracy plots

average_val_acc_history_2_4_6=np.load('average_val_acc_history_2_4_6.npy')
average_val_acc_history_2_4_22=np.load('average_val_acc_history_2_4_22.npy')
average_val_acc_history_2_6_30=np.load('average_val_acc_history_2_6_30.npy')
average_val_acc_history_2_10_40=np.load('average_val_acc_history_2_10_40.npy')
average_val_acc_history_7_16_30=np.load('average_val_acc_history_7_16_30.npy')



plt.plot(average_val_acc_history_2_4_22)
plt.plot(average_val_acc_history_2_6_30)
plt.plot(average_val_acc_history_7_16_30)


plt.title('Validation Accuracy Comparison with Different Dilations')
plt.xlabel('Epochs')
# plt.ylabel('Epochs')
plt.legend([ "Dilation 2-4-22", "Dilation 2-6-30",  "Dilation 7-16-30"], loc="upper left")

