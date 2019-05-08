import wfdb 
import numpy as np 

# features array:
#
# (0 = no = negative = front)
# (1 = yes = positive = end)
#
# 0     parity = number of previous deliveries,
# 1     maternal age,
# 2     previous abortions, 
# 3     weight, 
# 4     hypertension, 
# 5     diabetes, 
# 6     smoker 
# 7     bleeding first trimester, 
# 8     bleeding second trimester, 
# 9     funneling, 
# 10    placental position,

# 6+ minutes 
def get_features(records):
    """
    get the feature vector from the record
    string array to int array, if not, drop the record

    :param records: wfdb records to check
    :return: numpy arrays with all 15 features, and gestation period
    """
    features_set = []
    output_set = []
    canceled_index_list = []
    for (num, record) in enumerate(records):
        signals, fields = wfdb.rdsamp(record.record_name,pb_dir='tpehgdb/tpehgdb')
        features = np.empty(15, dtype=float)
        good_record_flag = True
        output = 1
        # index of features array
        i = 0
        for item in (fields['comments']):
            if "None" in item:
                # append canceled index
                canceled_index_list.append(num)
                good_record_flag = False
                break
            else:
                # good record
                if "Gestation" in item:
                    # output
                    output = (item.split()[1])
                if "Parity" in item:                
                    features[i] = (item.split()[1])
                    i += 1
                if "Age" in item:
                    features[i] = (item.split()[1])
                    i += 1
                if "Abortions" in item:
                    features[i] = (item.split()[1])
                    i += 1
                if "Weight" in item:
                    features[i] = (item.split()[1])
                    i += 1
                if "Hypertension" in item:
                    if "no" in item:
                        features[i] = 0
                    else:
                        features[i] = 1
                    i += 1
                if "Diabetes" in item:
                    if "no" in item:
                        features[i] = 0
                    else:
                        features[i] = 1
                    i += 1
                if "Smoker" in item:
                    if "no" in item:
                        features[i] = 0
                    else:
                        features[i] = 1
                    i += 1
                if "Bleeding_first_trimester" in item:
                    if "no" in item:
                        features[i] = 0
                    else:
                        features[i] = 1
                    i += 1
                if "Bleeding_second_trimester" in item:
                    if "no" in item:
                        features[i] = 0
                    else:
                        features[i] = 1
                    i += 1
                if "Funneling" in item:
                    if "negative" in item:
                        features[i] = 0
                    else:
                        features[i] = 1
                    i += 1
                if "Placental_position" in item:
                    if "front" in item:
                        features[i] = 0
                    else:
                        features[i] = 1      
                    i += 1      
        if (good_record_flag):
            features_set.append(features)
            output_set.append(output)


    # arrays should have the same number of rows (number of records kept)
    features_array = np.empty(shape=(len(features_set),11), dtype=float)
    outputs_array = np.empty(shape=(len(output_set), 1), dtype=float)
    # convert list to ndarray
    for (i, feature_array) in enumerate(features_set):
        features_array[i] = feature_array[:-4]
    for (i, output_array) in enumerate(output_set):
        outputs_array[i] = output_array
    return features_array, outputs_array, canceled_index_list

# test preprocessing
if __name__ == "__main__":
    # record1 = wfdb.rdrecord('tpehgdb/tpehg1132',pb_dir='tpehgdb/tpehgdb') # dropped
    record2 = wfdb.rdrecord('tpehgdb/tpehg1134',pb_dir='tpehgdb/tpehgdb')
    r = []
    # r.append(record1)
    r.append(record2)
    f, o, a = get_features(r)
    for item in np.nditer(f):
        print((item))
    print(o[0])
    # signals, fields = wfdb.rdsamp(record.record_name,pb_dir='tpehgdb/tpehgdb')
    # print((fields['comments']))   # list
        

    