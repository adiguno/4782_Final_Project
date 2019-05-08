import time
import numpy as np 
import pickle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import csv


def generate_best_model(features, labels):
    '''
    Generates 144 models until it saves the 10 models with above 80% accuracy
    Models have different combinations of the numbers of neurons in the hidden layers
    Following the 'rule of thumb':
        size of output < hidden neurons < size of inpput
        2 < x < 15 (3-14)
    Trained on randomly split data for each model

    :param features: input features
    :param labels: output labels
    '''
    i = 0
    for hidden1 in range(3,15,1):
        for hidden2 in range(3,15,1):
            model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(15, 1)),
                    tf.keras.layers.Dense(hidden1, activation=tf.nn.relu),
                    tf.keras.layers.Dense(hidden2, activation=tf.nn.relu),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(2, activation=tf.nn.softmax)])
            model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
            model.fit(x_train, y_train, epochs=300)
            los, acc = model.evaluate(x_test, y_test)
            if (acc > .8):
                model.save('{}.model'.format(i))
                i += 1
                if i > 9:
                    return
          
          
def sensitivity_and_specificity(predictions, y_test):
    '''
    Sensitivity = total positive prediction / total actual positives
    Specificity = total negative prediction / total actual negatives
    
    :param predictions: predictions of the model
    :param y_test: actual data
    :return: (sensitivity, specificity)
    '''
    # get y_test (real results)
    real_positive = []
    real_negative = []
    for i, real in enumerate(y_test):
        if real == 0:
            real_negative.append(i)
        else: 
            real_positive.append(i)
    # separate predictions
    pred_positive = []
    pred_negative = []
    for i, pred in enumerate(predictions):
        if np.argmax(pred) == 0:
            pred_negative.append(i)
        else:
            pred_positive.append(i)
    # calculate sensitivity
    right = 0
    for index in pred_positive:
        if index in real_positive:
            right += 1
    sensitivity = right / len(real_positive)

    # calculate specificity
    right = 0
    for index in pred_negative:
        if index in real_negative:
            right += 1
    specificity = right / len(real_negative)
    return sensitivity, specificity


def initial_decent_model():
    '''
    Initial model testing
    '''
    # output: preterm vs term
    # class_names = ['term', 'preterm']
    with open('SMOTE_features_labels.pkl', 'rb') as f:  
        features, labels = pickle.load(f)

    # reshape
    features = np.reshape(features, (300, 15,1) )
    labels = np.reshape(labels, (300,) )
    # print(features.shape)

    # SPLIT
    rand_state = 44
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2, random_state=rand_state)
  
    NAME = 'DECENT'
    # decent model
    # data shape = (number_of_records, rows, cols)
    # input_data_shape = (useable_record_number, 1, number_of_features)
    tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(15, 1)),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    # testing
    model.fit(x_train, y_train, epochs=300, callbacks=[tensorboard])
    predictions = model.predict(x_test)
    s, ss = sensitivity_and_specificity(predictions, y_test)
    print(s,ss)


if __name__ == "__main__":
    with open('SMOTE_features_labels.pkl', 'rb') as f:  
            features, labels = pickle.load(f)

    # reshape data
    features = np.reshape(features, (300, 15,1) )
    labels = np.reshape(labels, (300,) )

    # generate models
    # generate_best_model(features, labels)

    # # initial model screening
    # accuracy_models = []
    # for i in range(11):
    #     mod = tf.keras.models.load_model('{}.model'.format(i))
    #     # Name = 'model_{}'.format(i)
    #     # accu = []
    #     # for trial in range(20):
    #     #     x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
    #     #     _, acc = mod.evaluate(x_test, y_test)
    #     #     accu.append(acc)
    #     # avg_acc = sum(accu) / len(accu)
    #     # accuracy_models.append(avg_acc)
    #     print(mod.summary())
    # # print(accuracy_models)

    # graphs
    # (hidden layer 1, hidden layer 2)
    # models configuration
    a = [(6,8), (7,7), (7,8), (8,9), (9,4), (9,10), (9,13), (10,11), (11,10), (11,14)]
    i = 0    

    # get tensorboard graph of a newly trained model
    for tup in a:
        hid1, hid2 = tup
        NAME = 'model_{}'.format(i)
        tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2)
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(15, 1)),
                    tf.keras.layers.Dense(hid1, activation=tf.nn.relu),
                    tf.keras.layers.Dense(hid2, activation=tf.nn.relu),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(2, activation=tf.nn.softmax)])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])  
        acc_history = model.fit(x_train, y_train, epochs=300,
                                   callbacks=[tensorboard])
        i +=1
    
        acc = []

    # get accuracy, sensitivity, specificity of 30 trials of he 10 models
    avg_acc_out = []
    avg_sen_out = []
    avg_spe_out = []
    # 10 models, 30 trials
    for i in range(10):
        model = tf.keras.models.load_model('{}.model'.format(i))
        acc = []
        sen = []
        spe = []
        for i in range(30):
            x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2, random_state=i)
            # model.fit(x_train, y_train, epochs=300)
            loss, ac = model.evaluate(x_test, y_test)
            predictions = model.predict(x_test)
            sens, spec = sensitivity_and_specificity(predictions, y_test)
            acc.append(ac)
            sen.append(sens)
            spe.append(spec)
        avg_acc =  sum(acc)/len(acc)
        avg_sen =  sum(sen)/len(sen)
        avg_spe =  sum(spe)/len(spe)
        avg_acc_out.append(avg_acc)
        avg_sen_out.append(avg_sen)
        avg_spe_out.append(avg_spe)
    # Write acc, sen, spe to a csv
    with open('model_performaces.csv', mode='w') as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        employee_writer.writerow(['accuracy', 'sensitivity', 'specificity'])
        for index in range(10):
            employee_writer.writerow(['model_{}'.format(index), avg_acc_out[index], avg_sen_out[index], avg_spe_out[index]])

