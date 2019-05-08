import wfdb 
import numpy as np 
import pickle

def rdFVL(fvlFileName, canceled_index, channel_num):
    """
    Extract Root mean square, median frequency, peak frequency, sample entropy 

    :param fvlFileName: name of the fvl file
    :param canceled_index: index of the dropped records
    :param channel_num: channel fo the record
    :return: numpy array with 4 features
    """
    array_length = 300 - len(canceled_index)
    FrFtArray = np.zeros((array_length,4), dtype = float)
    with open(fvlFileName, "r") as f:
        i = 0
        j = 0
        for line in f:
            # seperate the columns
            fields = line.split("|")
            # had to include this if statement to bypass the line of dashes
            if len(fields) > 1:
                chan = fields[1]
                # only look at one channel for each patient
                if (str(channel_num)) in chan:
                    if j not in canceled_index:
                        FrFtArray[i][0] = float(fields[5])
                        FrFtArray[i][1] = float(fields[6])
                        FrFtArray[i][2] = float(fields[7])
                        FrFtArray[i][3] = float(fields[8])
                        i += 1
                    j += 1                
    return FrFtArray



# #test program
# if __name__ == "__main__":
#     # canceled_index no longer stored
#     with open('features_outputs.pkl', 'rb') as f:  
#         feastures, outputs, canceled_index = pickle.load(f)
#     fvlFile = "tpehgdb_features__filter_0.08_Hz-4.0_Hz.fvl"
#     #fvlFile = "tpehgdb_features__filter_0.3_Hz-3.0_Hz.fvl"
#     # print(len(canceled_index))
#     fourFeatures = rdFVL(fvlFile, canceled_index, 1)
    
#     # for ind in canceled_index:
#     #     print(type(ind))
#     print(fourFeatures)
#     print(fourFeatures.dtype)