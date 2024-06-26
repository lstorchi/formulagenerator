import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from sklearn.linear_model import LinearRegression

import generators

if __name__ == "__main__":
    
    name = "Name"

    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="input pki file ", \
            required=True, type=str)
    parser.add_argument("--formula", \
            help="Specify the formula to check", \
            required=True, default="")
    parser.add_argument("-i","--inputlabels", help="Specify label name and file comma separated string"+\
            "\n  \"filname.xlsx,labelcolumnname,sheetname\"", \
            required=True, type=str, default="")

    args = parser.parse_args()

    fname = args.file
    formula = args.formula

    sline = args.inputlabels.split(",")

    if len(sline) != 3:
        print("Error in ", args.inputlabels)
        exit(1)

    excelfile = sline[0]
    labelname = sline[1]
    sheetname = sline[2]
    
    df = pd.read_pickle(fname)
    
    x = df[formula].values
    
    data = pd.read_excel(excelfile, sheetname)
    y = data[labelname].values

    labels = []
    if name in data.columns:
        labels = data[name]
    else:
        for i in range(y.shape[0]):
            labels.append(str(i))
    
    print("Shape: ", y.shape, x.shape)
    
    regressor = LinearRegression()
    regressor.fit(x.reshape(-1,1), y)
    
    y_pred = regressor.predict(x.reshape(-1,1))

    print('Coefficients: %15.8f Intecept: %15.8f\n'%( \
          regressor.coef_[0], regressor.intercept_))
    
    plt.scatter(y_pred, y,  color='black')
    
    i = 0
    for x,y in zip(y_pred,y):
        label = labels[i]
    
        plt.annotate(label, # this is the text
                     (x,y), # this is the point to label
                     textcoords="offset points", # how to position the text
                     xytext=(0,10), # distance from text to points (x,y)
                     ha='center') # horizontal alignment can be left, right or center
    
        i += 1
    
    #plt.xticks(())
    #plt.yticks(())
    
    plt.title(sheetname + " " + str(regressor.coef_) + \
            " * " + formula + " + " + str(regressor.intercept_))
    
    plt.xlabel("Predicted values " + labelname)
    plt.ylabel("Real values " + labelname)
    
    plt.show()
