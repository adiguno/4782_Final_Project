import wfdb 
import numpy as np 
import csv
import pickle 
import pandas as pd 
import signalFeatures as sf 


'''
['Gestation 35', 
'Rectime 31.3', 
'Age 30', 
'Parity 0', 
'Abortions 0', 
'Weight 58', 
'Hypertension no', 
'Diabetes no', 
'Placental_position front', 
'Bleeding_first_trimester no', 
'Bleeding_second_trimester no', 
'Funneling negative', 
'Smoker no']
'''
FEATURES = ['Gestation', 'Age', 'Parity', 'Abortions', 'Weight', 'Hypertension', 'Diabetes', 'Placental_position', 'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker']


def check_data():
    '''
    check 
        features_labels.pkl
        actual_features_labels.pkl
    '''
    with open('actual_features_labels.pkl', 'rb') as data_file:
        features, labels = pickle.load(data_file)
        # print(features, labels)
        for feature in features:
            print(feature)
        
    pass


def get_records():
    """
    get all 300 records from the database
    :return: a list of all 300 records
    """
    record_list_file_name = "RECORDS.txt"
    records = []
    with open(record_list_file_name, "r") as f:
        for line in f:
            record_name = line[:-1]
            record = wfdb.rdrecord(record_name,pb_dir='tpehgdb/tpehgdb')
            records.append(record)
    # stores the records
    # with open('records.pkl', 'wb') as f: 
    #     all_records = get_records()
    #     print(len(all_records))
    #     pickle.dump(all_records,f)
    return records


def read_data():
    '''
    read the records from the pickle file
    '''
    file_name = 'records.pkl'
    records = []
    with open(file_name, 'rb') as f:  
        records = pickle.load(f)
        return records
        # print(len(records))
     

def parsing_data(records=None):
    '''
    Extract the features from records
    300 records => 300 x 12 arr
    '''
    full_record_arr = np.empty((0,0))
    FULL_RECORD_DIMENSION = (300,12)

    for record in records:
        patient_record_values_list = []
        _, fields = wfdb.rdsamp(record.record_name,pb_dir='tpehgdb/tpehgdb')
        for field in fields['comments'][2:]:
            field_key_val_list = field.split()
            patient_record_values_list.append(field_key_val_list[1])
        del patient_record_values_list[1]
        record_arr = np.asarray(patient_record_values_list)
        full_record_arr = np.append(full_record_arr, record_arr)

    # reshape the full record array
    full_record_arr = full_record_arr.reshape(FULL_RECORD_DIMENSION)
    # convert to pandas dataframe
    all_rec_df = pd.DataFrame(full_record_arr)
    # set column name to FEATURES
    all_rec_df.columns = FEATURES

    return all_rec_df


def export_csv(records_df):
    records_df.to_csv('all_patient_records.csv', index=False)
    print('done')


if __name__ == "__main__":
    # print(FEATURES)
    # records = read_data()
    # print('done reading, now parsing')
    # all_records = parsing_data(records=records)   
    # print('export to csv')
    # export_csv(records_df=all_records)

    fvlFile = "tpehgdb_features__filter_0.08_Hz-4.0_Hz.fvl"
    canceled_index = []
    fourFeatures = sf.rdFVL(fvlFile, canceled_index, 1)
    print(fourFeatures)
