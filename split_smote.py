import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from imblearn.over_sampling import SMOTE

import pickle
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def gen_model():
    shapes = [(9,4), (9,10)]
    models = []
    for shape in shapes:
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(15, 1)),
                    tf.keras.layers.Dense(shape[0], activation=tf.nn.relu),
                    tf.keras.layers.Dense(shape[1], activation=tf.nn.relu),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(2, activation=tf.nn.softmax)])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        models.append(model)
    return models
        
        
if __name__ == "__main__":

    # gets the features and outputs (labels = gestation periods)
    with open('actual_features_labels.pkl', 'rb') as f:  
        features, labels = pickle.load(f)

    # reshape data
    features = np.reshape(features, (169, 15,) )
    labels = np.reshape(labels, (169,) )
    print(features.shape)
    print(labels.shape)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2)

    preterm = 0
    for item in y_train:
        if item == 1:
            preterm +=1
    print(preterm)
    # SMOTE
    sm = SMOTE()
    SMOTE_features, SMOTE_labels = sm.fit_resample(x_train, y_train)
    # preterm = 0
    # term = 0
    # for item in SMOTE_labels:
    #     if item == 1:
    #         preterm +=1
    #     else:
    #         term +=1
    # print(preterm)
    # print(term)
    print(SMOTE_features.shape)
    print(SMOTE_labels.shape)
    depth, _ = SMOTE_features.shape

    reshaped_SMOTE_features = np.reshape(SMOTE_features, (depth, 15,1))

    d, x = x_test.shape
    reshaped_SMOTE_test = np.reshape(x_test, (d, 15,1))


    models = gen_model()
    i = 0
    for model in models:
        NAME = 'model_{}'.format(i)
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
        acc_history = model.fit(reshaped_SMOTE_features, SMOTE_labels, epochs=300, callbacks=[tensorboard])
        model.evaluate(reshaped_SMOTE_test, y_test)
        i +=1

