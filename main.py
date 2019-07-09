import wfdb 
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import numpy as np
from timeit import default_timer as timer
import preprocessing 
import pickle
import scipy
import signalFeatures
from imblearn.over_sampling import SMOTE
import tensorFlowTesting as tft
import csv

FEATURES = ['parity', 'maternal age', 'previous abortions', 'weight', 'hypertension', 'diabetes','smoker','bleeding first trimester', 'bleeding second trimester', 'funneling', 'placental position','Root mean square', 'median frequency', 'peak frequency', 'sample entropy','gestation period']

# 13 minutes op
def get_records():
    """
    get all 300 records from the database
    :return: a list of all 300 records
    """
    file_name = "RECORDS.txt"
    records = []
    with open(file_name, "r") as f:
        for line in f:
            record_name = line[:-1]
            record = wfdb.rdrecord(record_name,pb_dir='tpehgdb/tpehgdb')
            records.append(record)
    return records

def print_10(arr):
    for i in range(10):
        print(arr[i])

if __name__ == "__main__":
    # print option: 3 digits
    np.set_printoptions(precision=3,floatmode='fixed')

    # records = get_records()
    # # stores the records
    # with open('records.pkl', 'wb') as f: 
    #     pickle.dump(records,f)

    """records.pkl is too large to push to Git (~1Gb)"""
    # # gets the records
    # with open('records.pkl', 'rb') as f:  
    #     records = pickle.load(f)
    # # print(len(records))
    
    # # get features
    # medFeatures, labels, canceled_index = preprocessing.get_features(records)
    # fvlFile = "tpehgdb_features__filter_0.08_Hz-4.0_Hz.fvl"
    # fourFeatures = signalFeatures.rdFVL(fvlFile, canceled_index, 1)
    # # combine features
    # combined_features = np.append(medFeatures, fourFeatures, axis=1)
    # # stores the features
    # with open('features_labels.pkl', 'wb') as f: 
    #     pickle.dump([combined_features, labels],f)

    # gets the features and outputs (labels = gestation periods)
    with open('features_labels.pkl', 'rb') as f:  
        features, labels = pickle.load(f)
        
        with open('project_features_labels.csv', 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',')
            csv_writer.writerow([feature for feature in FEATURES])
            for row in range(len(features)):
                print(features[row])

        # print(features.shape)
    cout = 0
    for feature in features:
        cout+=1
    print(cout)
#     # preterm = 1
#     # term = 0
#     # formatting the labels
#     for x in np.nditer(labels, op_flags=['readwrite']):
#         if x < 37:
#             x[...] = 1
#         else: 
#             x[...] = 0
#     # label is converted to int and falttened
#     labels = labels.astype(int)
#     labels = labels.flatten()
#     print(labels.shape)
#     print(features[:5])
#     print(labels[:5])
# #     # gets the features and outputs (labels = gestation periods)
# #     with open('actual_features_labels.pkl', 'wb') as f:  
# #         pickle.dump([features, labels], f)

#     # SMOTE
#     sm = SMOTE()
#     SMOTE_features, SMOTE_labels = sm.fit_resample(features, labels)
#     with open('SMOTE_features_labels.pkl', 'wb') as f: 
#         pickle.dump([SMOTE_features, SMOTE_labels],f)
    # print_10(SMOTE_features)
    # print_10(SMOTE_labels)

    # # get SMOTE data
    # with open('SMOTE_features_labels.pkl', 'rb') as f:  
    #     features, labels = pickle.load(f)
    # # reshape data
    # features = np.reshape(features, (300, 15,1) )
    # labels = np.reshape(labels, (300,) )

    # # trials
    # # splitting the data
    # acc = []
    # sen = []
    # spe = []

    # # load the model
    # model = tf.keras.models.load_model('6.model')
    # # gets the sensitivity, accuracy, specificity of each trial
    # for i in range(30):
    #     x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=.2, random_state=i)
    #     # model.fit(x_train, y_train, epochs=300)
    #     loss, ac = model.evaluate(x_test, y_test)
    #     predictions = model.predict(x_test)
    #     sens, spec = tft.sensitivity_and_specificity(predictions, y_test)
    #     acc.append(ac)
    #     sen.append(sens)
    #     spe.append(spec)
    # # write sen and spe to csv file
    # with open('sensitivity_specificity.csv', mode='w') as employee_file:
    #     employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     employee_writer.writerow(['sensitivity', 'specificity'])
    #     for index, _ in enumerate(spe):
    #         employee_writer.writerow([sen[index], spe[index]])
    # # print all stats
    # print("avg acc: {}".format(sum(acc)/len(acc)))          # avg: 0.6811111082633337
    # print("avg sensitivity: {}".format(sum(sen)/len(sen)))  # avg: 0.7544749259299052
    # print("avg specificity: {}".format(sum(spe)/len(spe)))  # avg: 0.6658020327534167

    
