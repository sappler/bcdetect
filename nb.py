import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.model_selection import train_test_split

import os

#diagnose cancer based on physical characteristics of tissue
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)

def get_train_test_split(raw, per):
    x = raw.iloc[:,1:31]
    x['diagnosis'] = (x['diagnosis'] == 'M').astype(float)
    x_train, x_test = train_test_split(x, test_size=per)
    return x_train, x_test


def create_model(my_learning_rate, feature_layer, my_metrics):
  model = tf.keras.models.Sequential()
  model.add(feature_layer)
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,),
                                  activation=tf.sigmoid),)
  model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),loss=tf.keras.losses.BinaryCrossentropy(),metrics=my_metrics)

  return model        


def train_model(model, dataset, label,epochs, bsize):
  features = {name:np.array(value) for name, value in dataset.items()}
  history = model.fit(x=features, y=label, batch_size=bsize,epochs=epochs)
  return history

raw = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
test_per = .2
raw.columns = raw.columns.str.replace(' ', '_', regex=True)
x_train, x_test = get_train_test_split(raw,test_per)

y_train  = np.array(x_train.pop('diagnosis')) 
y_test  = np.array(x_test.pop('diagnosis')) 
#print(y_train)
x_train_norm = (x_train - x_train.mean()) / x_train.std()
x_test_norm = (x_test - x_test.mean()) / x_test.std()
fols = [tf.feature_column.numeric_column(name) for name in list(x_train_norm.columns)[1:]]
flayer = tf.keras.layers.DenseFeatures(fols)

lerate = .001
epochs = 50
batch_size = 10
classification_threshold = .40
val_split = 0.35

METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy', 
                                      threshold=classification_threshold),
      tf.keras.metrics.Precision(thresholds=classification_threshold,
                                 name='precision' 
                                 ),
      tf.keras.metrics.Recall(thresholds=classification_threshold,
                              name="recall"),
      tf.keras.metrics.AUC(num_thresholds=100, name='auc'),
]

mod = create_model(lerate, flayer,METRICS)

history = train_model(mod, x_train_norm, y_train, epochs, bsize)
mod.summary()

features = {name:np.array(value) for name, value in x_test_norm.items()}

mod.evaluate(x = features, y = y_test, batch_size=bsize)
