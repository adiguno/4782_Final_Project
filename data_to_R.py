import numpy as np 
import csv
import pickle 
import wfdb 


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
    return records


def read_data():

    pass


def export_csv():
    pass


if __name__ == "__main__":
        # gets the records
    with open('records.pkl', 'wb') as f:  
        records = pickle.load(f)
    # print(len(records))
