import pickle as pk
from shutil import rmtree
from os import listdir, makedirs
import numpy as np

def restructure():
    data_path = 'network_data_2/'
    all_files = listdir(data_path)
    # clear up the new path for newly formatted data and also the maximum used to normalize
    new_path = 'network_data_rest_2/'
    rmtree(new_path)
    makedirs(new_path)
    #maximum_path = 'normalize_maximum/'
    #rmtree(maximum_path)
    #makedirs(maximum_path)
    #number_classes = 'number_classes/'
    #rmtree(number_classes)
    #makedirs(number_classes)
    # the previous classification
    prev = 'k'
    out = 0
    i=0
    maximum_all = 0
    num_classes = 10
    num_files_per_class = 100000
    for file in all_files:
        if int(file.split('_')[0]) <= num_classes:
            # check if it's the same classification, if so keep running on the number counter
            if prev == file.split('_')[0]:
                out = out + i + 1
            else:
                out = 0

            prev = file.split('_')[0]
            input_data = np.fromfile(data_path + file, dtype="float32")
            maximum_input = max(abs(input_data))
            maximum_all = max(maximum_input, maximum_all)
            I_data = input_data[::2]
            Q_data = input_data[1::2]
            input_data = np.vstack((I_data, Q_data))
            leftover_data = len(Q_data) - (len(Q_data) % (8192))
            input_data = np.moveaxis(np.reshape(input_data[:, :leftover_data], (2, 8192, -1), 'F'), -1, 0)
            
            for i, frame in enumerate(input_data):
                if i + out > num_files_per_class:
                    break
                with open('network_data_rest_2/' + file.split('_')[0] + '_' + str(i+ out).zfill(8), "bx") as fd:
                    pk.dump(frame, fd)
    #with open(maximum_path + 'maximum', 'bx') as fd:
    #    pk.dump(maximum_all, fd)
    #with open(number_classes + 'number_classes', 'bx') as fd:
    #    pk.dump(num_classes, fd)


if __name__ == "__main__":
    restructure()
