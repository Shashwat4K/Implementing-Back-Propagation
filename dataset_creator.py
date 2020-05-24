import numpy as np
import pandas as pd 
from os.path import join

def dataset_creator(): 
    rows = int(input("Enter no. of training examples you want: "))
    train_size = int(input("Enter size of input vector: "))
    test_size = int(input("Enter size of target output vector: "))
    cols = train_size + test_size
    data_block = np.random.choice([0, 1], size=(rows, cols), p=[0.5, 0.5]) 
    print("Your data block has been created, first {} columns are training examples, last {} columns are target outputs".format(train_size, test_size))
    done = False 
    file_name = None
    while done != True:
        file_name = input("Enter name of the file you want to save it as: ")
        if len(file_name) != 0:
            done = True
        else:
            print("Try again, enter valid path!")

    df = pd.DataFrame(data_block)
    df.to_csv(join('Input', file_name+".csv"))
    print("File name {}.csv is saved in 'Input' directory".format(file_name))           

if __name__ == "__main__":
    dataset_creator()