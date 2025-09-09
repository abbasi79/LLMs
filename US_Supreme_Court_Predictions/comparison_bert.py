from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import  confusion_matrix,ConfusionMatrixDisplay

from confusion_mat import *


# Load pretrained model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


X_test_np=X_test.to_numpy()
# predictions_all=np.zeros(len(X_test))
sigmoid_activation = torch.nn.Sigmoid()

indices_win = np.where(y_test  == 0)
X_test_np= X_test_np[indices_win]
y_test= y_test.to_numpy()[indices_win]
predictions_all=np.zeros(len(X_test_np))

for i in range(len(X_test_np)):
    
    print(f' Case # {i}')
# Perform inference
    inputs = tokenizer(X_test_np[i], return_tensors="pt",  max_length=500)

    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class
    # predictions = sigmoid_activation(outputs.logits, dim=-1)
    predictions = sigmoid_activation(outputs.logits)
    binary_output = torch.argmax((predictions  >= 0.5).int()).numpy()

    predictions_all[i]=binary_output


test_oyez=y_test.to_numpy()

accuracy = np.mean(test_oyez == predictions_all)

# %% confusion matrices

cm = confusion_matrix(test_oyez, predictions_all)
make_confusion_matrix(cm)






