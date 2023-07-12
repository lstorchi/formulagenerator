import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append("./common/")

import generators

if __name__ == "__main__":

    quiet = True
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="input pki file ", \
            required=True, type=str)
    parser.add_argument("-o","--output", help="output csv file [default=feature_mse.csv]", \
            required=False, type=str, default="feature_mse.csv")
    parser.add_argument("-n","--numofiterations", help="Number of LR iterations ", \
            required=False, type=int, default=1000)
    parser.add_argument("-i","--inputlabels", help="Specify label name, file and sheetname comma separated string"+\
            "\n  \"filname.xlsx,labelcolumnname,sheetname\"", \
            required=False, type=str, default="")
    parser.add_argument("-s", "--split", \
            help="Split by a key [default=\"\"]", required=False, default="")
    parser.add_argument("-u", "--usefullsetrmse", \
            help="Sort using the fullset RMSE instead of the random splitting", action="store_true", default=False)
    
    args = parser.parse_args()
    
    filename = args.file

    df = pd.read_pickle(filename)
    N = df.shape[0]

    DE_Array = None
    if not quiet:
        print(args.inputlabels)
        
    if args.inputlabels == "":
        splitted = generators.defaultdevalues.split(",")
        DE_array = np.array(np.float64(splitted)).reshape(N, 1)

        if df.shape[0] != len(splitted):
                print("Labels and features dimension are not compatible")
                exit(1)
    else:
        sline = args.inputlabels.split(",")
        if len(sline) != 3:
            print("Error in --inputlabels option")
            exit(1)
        
        data = pd.read_excel(sline[0], sline[2])

        if args.split != "":
            ssplit = args.split.split(";")
            if (len(ssplit) != 2):
                print("Error in split option ", args.split, " must have key;value pair")
                exit(1)
 
            key = ssplit[0]
            value = ssplit[1]
 
            if not (key in data.columns):
                print("Error in split option ", key, " not present ")
                exit(1)
 
            uniqvalues  = set(data[key].values)
 
            if not int(value) in uniqvalues:
                print("Error value ", int(value), " not present")
                exit(1)
 
            if not quiet:
                print("All possible value are:")
                for v in uniqvalues:
                    print("  ",v)
 
            isthevalue =  data[key] == int(value)
            data = data[isthevalue]

            if not quiet:
                print("Selected data: ")
                print(data.shape)
 
        label = data[sline[1]]
        DE_array = label.values

    fname = "finalselectedformulas.txt"
    fp = open(fname, "w")

    if not quiet:
        print("I will process :", len(df.columns), " features " )

    feature_mse_dataframe = None

    if args.usefullsetrmse:
        feature_mse_dataframe = \
            generators.feature_check_fullsetrmse (range(0, len(df.columns)), 
                df, DE_array)

        minvalue_lr = np.min(feature_mse_dataframe['rmse'].values)
        bestformula_lr = \
            feature_mse_dataframe[feature_mse_dataframe['rmse'] \
            == minvalue_lr]['formulas'].values[0]

        print(" Min RMSE value , ", minvalue_lr, " , Related formula , ", bestformula_lr)

        fp.write(bestformula_lr + "\n")
    else:
        feature_mse_dataframe = \
            generators.feature_check_lr (range(0, len(df.columns)), df, \
            DE_array, args.numofiterations)

        minvalue_lr = np.min(feature_mse_dataframe['mse'].values)
        bestformula_lr = \
            feature_mse_dataframe[feature_mse_dataframe['mse'] \
            == minvalue_lr]['formulas'].values[0]

        print(" Min LR value , ", minvalue_lr, " ,  Related formula , ", bestformula_lr)

        fp.write(bestformula_lr + "\n")

    feature_mse_dataframe.to_csv(args.output)

    pearson_max = np.max(feature_mse_dataframe['percoeff'].values)
    bestformula_pearson = \
            feature_mse_dataframe[feature_mse_dataframe['percoeff'] \
            == pearson_max]['formulas'].values[0]

    print(" Max Pearson R value , ", pearson_max, " ,  Related formula , ", bestformula_pearson)

    fp.write(bestformula_lr + "\n")

    r2_max = np.max(feature_mse_dataframe['r2'].values)
    bestformula_r2 = \
            feature_mse_dataframe[feature_mse_dataframe['r2'] \
            == r2_max]['formulas'].values[0]

    print(" Max R2 value , ", r2_max, " ,  Related formula , ", bestformula_r2)

    fp.write(bestformula_lr + "\n")

    """
    pval_min = np.min(feature_mse_dataframe['pval'].values)
    bestformula_pval = \
            feature_mse_dataframe[feature_mse_dataframe['pval'] \
            == pval_min]['formulas'].values[0]

    print("")
    print(" Min P-Value value: ", pval_min)
    print("   Related formula: ", bestformula_pval)

    fp.write(bestformula_pval + "\n")
    """

    fp.close()
