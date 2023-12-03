import numpy as np
import pandas as pd

from datetime import datetime

# In[ ]:

# load feature data
    
import pickle

with open('feature_log-mel_df.pickle','rb') as f:
    dataset_df = pickle.load(f)

# In[ ]:
    
# split data
    
from sklearn.model_selection import train_test_split

# Split the dataset
X = np.array(dataset_df['feature'].tolist())
y = np.array(dataset_df['label'].tolist())

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, random_state=1, stratify=Y_test)

# In[ ]:

# Add dimension for the channel

X_train = X_train.reshape(-1,42,173,1)
X_val = X_val.reshape(-1,42,173,1)
X_test = X_test.reshape(-1,42,173,1)

print(X_train.shape, X_val.shape, X_test.shape)

# In[ ]:

# FCN Model

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

model = Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(Conv2D(filters=16, kernel_size=(2, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 3)))
model.add(Conv2D(filters=32, kernel_size=(2, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(2, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=128, kernel_size=(2, 4), activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(rate=0.5))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.summary()

# In[ ]:

# transfer learning

# import tensorflow as tf
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.applications import MobileNet

# pre_trained_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,173,3))
# pre_trained_model.summary()

# model = Sequential()
# model.add(pre_trained_model)
# model.add(GlobalAveragePooling2D())
# model.add(Dense(64,activation='relu'))
# model.add(Dropout(0.5))                          # 0.25 오버피팅 발생함
# model.add(Dense(10, activation='softmax'))

# model.compile(loss='categorical_crossentropy', 
#             optimizer=tf.keras.optimizers.Adam(2e-5), metrics=['acc'])
# model.summary()

# In[ ]:
    
# callbacks    

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping    

checkpoint = ModelCheckpoint(filepath='best_fcn.hdf5', # filename
                                monitor='val_accuracy',   # val_accuracy 값이 개선되었을때 호출
                                verbose=1,                # 로그 출력 
                                save_best_only=True,      # 가장 best 값 저장
                                mode='auto')              # 알아서 best 찾음

earlystopping = EarlyStopping(monitor='val_loss',  # 모니터 기준 설정 (val loss) 
                               patience=5,        # 3회 Epoch동안 개선되지 않는다면 종료
                               verbose=1
                              )

# checkpointer = ModelCheckpoint(filepath='best_fcn.hdf5', 
#                                monitor='val_accuracy', verbose=1, 
#                                save_best_only=True)

#callbacks = [checkpoint, earlystopping]
callbacks = [checkpoint]

# In[ ]:
    
# run model

epochs = 150
batch_size = 256

hist = model.fit(X_train, Y_train,
              batch_size=batch_size, epochs=epochs,
              validation_data=(X_val, Y_val), callbacks=callbacks,
              verbose=1)

# In[ ]:
    
# Evaluating the model on the training and testing set    

best_model = load_model('best_fcn.hdf5')
score = best_model.evaluate(X_train, Y_train, verbose=0)
print("Training Accuracy: ", score[1])
score = best_model.evaluate(X_val, Y_val, verbose=0)
print("Validation Accuracy: ", score[1])
score = best_model.evaluate(X_test, Y_test, verbose=0)
print("Testing Accuracy: ", score[1])

# In[ ]:
    
# plot trend of accuracy & loss    
    
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14,6))
ax1 = plt.subplot(1,2,1)
ax1.plot(hist.history['accuracy'], label='train')
ax1.plot(hist.history['val_accuracy'], label='validation')
ax1.set_title('Accuracy Trend')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(loc='best')
ax1.grid()
ax2 = plt.subplot(1,2,2)
ax2.plot(hist.history['loss'], label='train')
ax2.plot(hist.history['val_loss'], label='validation')
ax2.set_title('Loss Trend')
ax2.set_ylabel('loss')
ax2.set_xlabel('epoch')
ax2.legend(loc='best')
ax2.grid()
plt.show()


# In[ ]:

# plot confusion matrix
    
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

Y_pred = best_model.predict(X_test)
cm = confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))

class_dic = {3: 'dog_bark', 2: 'children_playing', 1: 'car_horn', 
              0: 'air_conditioner', 9: 'street_music', 6: 'gun_shot', 
              8: 'siren', 5: 'engine_idling', 7: 'jackhammer', 4: 'drilling'}
classes = [class_dic[key] for key in sorted(class_dic.keys())]

normalize = True

if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

df_cm = pd.DataFrame(cm, index =[i for i in classes],columns=[i for i in classes])

plt.figure(figsize=(16,12))
plt.title('Confusion Matrix')
sns.heatmap(df_cm,annot=True)


# In[ ]:
    
# Plot a confusion matrix

from sklearn import metrics

Y_pred = best_model.predict(X_test)
matrix = metrics.confusion_matrix(Y_test.argmax(axis=1), Y_pred.argmax(axis=1))

# Confusion matrix code (from https://github.com/triagemd/keras-eval/blob/master/keras_eval/visualizer.py)
def plot_confusion_matrix(cm, concepts, normalize=False, show_text=True, fontsize=18, figsize=(16, 12),
                          cmap=plt.cm.coolwarm_r, save_path=None, show_labels=True):
 
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError('Invalid confusion matrix shape, it should be square and ndim=2')

    if cm.shape[0] != len(concepts) or cm.shape[1] != len(concepts):
        raise ValueError('Number of concepts (%i) and dimensions of confusion matrix do not coincide (%i, %i)' %
                          (len(concepts), cm.shape[0], cm.shape[1]))

    plt.rcParams.update({'font.size': fontsize})

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:
        cm = cm_normalized
        print(cm)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, vmin=np.min(cm), vmax=np.max(cm), alpha=0.8, cmap=cmap)

    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    plt.ylabel('True label', fontweight='bold')
    plt.xlabel('Predicted label', fontweight='bold')

    if show_labels:
        n_labels = len(concepts)
        ax.set_xticklabels(concepts)
        ax.set_yticklabels(concepts)
#         plt.xticks(np.arange(0, n_labels, 1.0), rotation='vertical')
        plt.xticks(np.arange(0, n_labels, 1.0), rotation=45)
        plt.yticks(np.arange(0, n_labels, 1.0))
    else:
        plt.axis('off')

    fmt = '.2f' if normalize else 'd'
    if show_text:
        # http://stackoverflow.com/questions/21712047/matplotlib-imshow-matshow-display-values-on-plot
        min_val, max_val = 0, len(concepts)
        ind_array = np.arange(min_val, max_val, 1.0)
        x, y = np.meshgrid(ind_array, ind_array)
        for i, (x_val, y_val) in enumerate(zip(x.flatten(), y.flatten())):
            c = cm[int(x_val), int(y_val)]
            ax.text(y_val, x_val, format(c, fmt), va='center', ha='center')

    if save_path is not None:
        plt.savefig(save_path)

class_dictionary = {3: 'dog_bark', 2: 'children_playing', 1: 'car_horn', 0: 'air_conditioner', 9: 'street_music', 6: 'gun_shot', 8: 'siren', 5: 'engine_idling', 7: 'jackhammer', 4: 'drilling'}
classes = [class_dictionary[key] for key in sorted(class_dictionary.keys())]

plot_confusion_matrix(matrix, classes)    


