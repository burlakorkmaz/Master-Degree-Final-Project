import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
import scikitplot as skplt
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
model = load_model("models\\model_vgg.h5")

#%%
images_list = []
counter = 1
for file_name in file_names:
    path =  data_folder + file_name
    image_array = cv2.imread(path)
    image_array = cv2.resize(image_array, (img_size, img_size))
    image_array = preprocess_input(image_array)
    images_list.append(image_array)
    if counter % 100 == 0:
        print(str(counter) + " image loaded...")
    counter += 1
    
images_array = np.array(images_list, dtype=np.float32)

predictions = model.predict(images_array, batch_size = batch_size, verbose = True)
#%%

test_labels = to_categorical(labels, num_classes=2)
loss, acc = model.evaluate(images_array, test_labels, verbose = True)
#%%    
#ROC curves
y_true = labels     
skplt.metrics.plot_roc_curve(y_true, predictions)
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