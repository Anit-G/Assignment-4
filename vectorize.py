from MFCC import Extract_Feature_Vec
import numpy as np
import os
import pickle
import logging as log
log.basicConfig(filename="vec.log",filemode='w',format='%(levelname)s - %(message)s',level=log.DEBUG)

def vectorize(path):
    fv,_,_ = Extract_Feature_Vec(path)
    return fv

def vectorize_files(file_pth):
    Data = {}
    for pth in file_pth:
        log.debug(f"---FILE PATH: {pth}---")
        file_data = {}
        for name in os.listdir(pth):
            sample_path = os.path.join(pth,name)
            vec = vectorize(sample_path)
            file_data[name] = vec
            log.debug(f"File Name: {name}\n\t\t\tVector: {vec.shape}")

        a = pth.split('/')
        a = a[-2]+"_"+a[-1]
        Data[a] = file_data
        log.debug(f"---SAVING DATA for file: {a}----")
    return Data

if __name__ == "__main__":
    Dev_pth = ["./Data/Dev/Team-07"]
    iso_pths = ["./Data/Isolated_Digits/2/dev","./Data/Isolated_Digits/2/train",
                "./Data/Isolated_Digits/3/dev","./Data/Isolated_Digits/3/train",
                "./Data/Isolated_Digits/4/dev","./Data/Isolated_Digits/4/train",
                "./Data/Isolated_Digits/5/dev","./Data/Isolated_Digits/5/train",
                "./Data/Isolated_Digits/z/dev","./Data/Isolated_Digits/z/train",]
    data = vectorize_files(iso_pths)

    # saving data as a pickle file
    file_handle = open('Team7_data.txt','wb')
    pickle.dump(data,file_handle)

    
    
