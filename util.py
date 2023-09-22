import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def generateSequence(path, filename):
    gRNA = pd.read_csv(path + filename)
    protospacer = gRNA['protospacer'].values

    # Extract positions
    a = len(protospacer[0])
    position = pd.DataFrame([[char for char in spacer] for spacer in protospacer])

    position.columns = [f"position_{i+1}" for i in range(a)]

    position2_array = np.array([protospacer[i][j:j+2] for i in range(len(protospacer)) for j in range(a-1)]).reshape(len(protospacer), a-1)
    position2 = pd.DataFrame(position2_array)
    position2.columns = [f"position_{i+1}" for i in range(a-1)]

    # Define categories
    mononucleotide_categories = [['A', 'C', 'G', 'T']] * a
    dinucleotide_categories = [['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']] * (a-1)

    # One-hot encoding for mononucleotide features
    encoder_mono = OneHotEncoder(categories=mononucleotide_categories, sparse=False)
    newdata = pd.DataFrame(encoder_mono.fit_transform(position))
    newdata.columns = [f"position_{i+1}_{category}" for i, categories in enumerate(encoder_mono.categories_) for category in categories]

    # One-hot encoding for dinucleotide features
    encoder_di = OneHotEncoder(categories=dinucleotide_categories, sparse=False)
    newdata2 = pd.DataFrame(encoder_di.fit_transform(position2))
    newdata2.columns = [f"position_{i+1}_{category}" for i, categories in enumerate(encoder_di.categories_) for category in categories]

    # Dinucleotide counts
    dinucleotide1 = ['A','T','C','G']
    count1 = pd.DataFrame([[spacer.count(d) for d in dinucleotide1] for spacer in protospacer])
    count1.columns = [f"{d}count" for d in dinucleotide1]

    dinucleotide2 = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    count2 = pd.DataFrame([[spacer.count(d) for d in dinucleotide2] for spacer in protospacer])
    count2.columns = [f"{d}count" for d in dinucleotide2]

    poi_feat_train = pd.concat([newdata, newdata2, count1, count2], axis=1)
    poi_feat_train.to_csv(path + "data_poiFeat.csv", index=False)

def preprocess_seq(data):
    print("Start preprocessing the sequence")
    length = len(data[0])
    DATA_X = np.zeros((len(data),4,length), dtype=int)
    print(DATA_X.shape)
    for l in range(len(data)):
        if l % 10000 == 0:
            print(str(l) + " sequences processed")
        for i in range(length):
            try: data[l][i]
            except: print(data[l], i, length, len(data))
            if data[l][i]in "Aa":
                DATA_X[l, 0, i] = 1
            elif data[l][i] in "Cc":
                DATA_X[l, 1, i] = 1
            elif data[l][i] in "Gg":
                DATA_X[l, 2, i] = 1
            elif data[l][i] in "Tt":
                DATA_X[l, 3, i] = 1
            else:
                print("Non-ATGC character " + data[i])
                sys.exit()
    print("Preprocessing the sequence done")
    return DATA_X
