import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.utils import to_categorical
#import scikitplot as skplt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sn


img_size = 224
batch_size = 32
data_folder = "Dataset\\dolphin_signal_dataset\\Test\\"
test_csv_path = 'Dataset\\dolphin_signal_dataset\\dolphin_signal_test.csv'
    

test_df = pd.read_csv(test_csv_path)
file_names = test_df['file_names'].values.tolist()
labels = np.array(test_df['labels'].values.tolist(), dtype=np.float64)

#%%
model = load_model("models\\model_cnn.h5")

#%%
images_list = []
counter = 1
for file_name in file_names:
    path =  data_folder + file_name
    image_array = cv2.imread(path, 0)
    image_array = cv2.resize(image_array, (img_size, img_size))
    images_list.append(image_array)
    if counter % 100 == 0:
        print(str(counter) + " image loaded...")
    counter += 1

images_list =  [x * (1./255) for x in images_list]    
images_array = np.array(images_list, dtype=np.float32)

predictions = model.predict(images_array, batch_size = batch_size, verbose = True)
#%%

test_labels = to_categorical(labels, num_classes=2)
loss, acc = model.evaluate(images_array, test_labels, verbose = True)
#%%    
#ROC curves
y_true = labels    
# skplt.metrics.plot_roc_curve(y_true, predictions)
# plt.show()

fpr, tpr, _ = roc_curve(y_true.astype(int), predictions[:, 1], pos_label = 1)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC (AUC = %0.2f)' % (roc_auc))
plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()

#%%
#confusion matrix
y_probs = []
for i in range(len(labels)):
    if predictions[i][0] > predictions[i][1]:
        y_probs.append(0)
    elif predictions[i][0] < predictions[i][1]:
        y_probs.append(1)
        
matrix = confusion_matrix(y_true, y_probs) 
plt.figure(figsize = (10,7))
ax = sn.heatmap(matrix, annot=True)
ax.set(xlabel='predicted label', ylabel='true label')

#%%