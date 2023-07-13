import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.append("./common/")

import generators

if __name__ == "__main__":
    name = "Name"
    tabtoread = "nonxtb"

    quiet = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="input pki file ", \
            required=True, type=str)
    parser.add_argument("--formula", \
            help="Specify the formula to optimize", \
            required=True, default="")
    parser.add_argument("-i","--inputlabels", help="Specify label name and file comma separated string"+\
            "\n  \"filname.xlsx,labelcolumnname,sheetname\"", \
            required=True, type=str, default="")
    parser.add_argument("--fourelementsformula", \
            help="Generate formulas including four elements", action="store_true",
            required=False, default=False)

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
    if not quiet:
        print(data.columns)
    if name in data.columns:
        labels = data[name]
    else:
        for i in range(y.shape[0]):
            labels.append(str(i))

    if not quiet:
        print("Shape: ", y.shape, x.shape)
    
    regressor = LinearRegression()
    regressor.fit(x.reshape(-1, 1), y)
    
    y_pred = regressor.predict(x.reshape(-1,1))
    bestr2 = r2_score (y, y_pred)
    bestformula = formula
    best_y_pred = y_pred
    best_regressor = regressor

    if not quiet:
        print("%10.5f "%(bestr2), bestformula)
        print('Coefficients: %15.8f Intecept: %15.8f\n'%( \
              regressor.coef_[0], regressor.intercept_))
    
    num = formula.split("/")[0]
    denum = formula.split("/")[1]

    if args.fourelementsformula:
        firstnum = ""
        secondnum = ""
        operator = ""

        dfirstnum = ""
        dsecondnum = ""
        doperator = ""

        if len(num.split("+")) == 2: 
            operator = "+"
            firstnum = num.split("+")[0][1:]
            secondnum = num.split("+")[1][:-1]
        elif len(num.split("-")) == 2: 
            operator = "-"
            firstnum = num.split("-")[0][1:]
            secondnum = num.split("-")[1][:-1]
        else:
            print("Error in formula shape")
            exit(1)

        if len(denum.split("+")) == 2: 
            doperator = "+"
            dfirstnum = denum.split("+")[0][1:]
            dsecondnum = denum.split("+")[1][:-1]
        elif len(denum.split("-")) == 2: 
            doperator = "-"
            dfirstnum = denum.split("-")[0][1:]
            dsecondnum = denum.split("-")[1][:-1]
        else:
            print("Error in formula shape")
            exit(1)
 
        if not quiet:
            print("Check Formula elements: |", \
                firstnum, "|", secondnum, "|", \
                dfirstnum, "|", dsecondnum )
        nstep = 10

        a = np.float64(np.float64(1.0)/(np.float64(nstep)))
        for ai in range(0, nstep):
            b =  np.float64(np.float64(1.0)/(np.float64(nstep)))
            for bi in range(0, nstep):
                c = np.float64(np.float64(1.0)/(np.float64(nstep)))
                for ci in range(0, nstep):
                    d = np.float64(np.float64(1.0)/(np.float64(nstep)))
                    for di in range(0, nstep):
                        newf = \
                            "(("+str(a)+"*("+firstnum+"))"+ \
                                operator+\
                            " ("+str(b)+"*("+secondnum+")))" + "/" + \
                            "(("+str(c)+"*("+dfirstnum+"))"+ \
                                doperator+\
                             " ("+str(d)+"*("+dsecondnum+")))"

                        #print(newf)

                        newx = generators.get_new_feature(data, newf)
                        newx = np.asarray(newx)
                        regressor = LinearRegression()
                        regressor.fit(newx.reshape(-1, 1), y)
                        y_pred = regressor.predict(newx.reshape(-1,1))

                        r2v = r2_score (y, y_pred)

                        if (r2v > bestr2):
                            bestr2 = r2v 
                            bestformula = newf
                            best_y_pred = y_pred
                            best_regressor = regressor

                        #print("%10.5f "%(r2v), newf)
                        d += np.float64(np.float64(1.0)/(np.float64(nstep)))
                    c += np.float64(np.float64(1.0)/(np.float64(nstep)))
                b += 1.0/(np.float64(nstep))
            a += 1.0/(np.float64(nstep))
    else:
        firstnum = ""
        secondnum = ""
        operator = ""

        if len(num.split("+")) == 2: 
            operator = "+"
            firstnum = num.split("+")[0][1:]
            secondnum = num.split("+")[1][:-1]
        elif len(num.split("-")) == 2: 
            operator = "-"
            firstnum = num.split("-")[0][1:]
            secondnum = num.split("-")[1][:-1]
        else:
            print("Error in formula shape")
            exit(1)

        if not quiet:
            print("Check Formula elements: |", \
                firstnum, "|", secondnum, "|", denum, "|")

        nstep = 10
        a = np.float64(np.float64(1.0)/(np.float64(nstep)))
        for ai in range(0, nstep):
            b =  np.float64(np.float64(1.0)/(np.float64(nstep)))
            for bi in range(0, nstep):
                c = np.float64(np.float64(1.0)/(np.float64(nstep)))
                for ci in range(0, nstep):
                    newf = \
                    "(("+str(a)+"*("+firstnum+"))"+ \
                            operator+\
                    " ("+str(b)+"*("+secondnum+")))" + "/" + \
                    "("+str(c)+"*("+denum+"))"

                    newx = generators.get_new_feature(data, newf)
                    newx = np.asarray(newx)
                    regressor = LinearRegression()
                    regressor.fit(newx.reshape(-1, 1), y)
                    y_pred = regressor.predict(newx.reshape(-1,1))

                    r2v = r2_score (y, y_pred)

                    if (r2v > bestr2):
                        bestr2 = r2v 
                        bestformula = newf
                        best_y_pred = y_pred
                        best_regressor = regressor

                    #print("%10.5f "%(r2v), newf)

                    c += np.float64(np.float64(1.0)/(np.float64(nstep)))
                b += 1.0/(np.float64(nstep))
            a += 1.0/(np.float64(nstep))

    if not quiet:
        print("%10.5f "%(bestr2), bestformula)
        print('Coefficients: %15.8f Intercept: %15.8f\n'%( \
            best_regressor.coef_[0], best_regressor.intercept_))
    else:
         print("%10.5f "%(bestr2), " , " best_regressor.intercept_, " + (",  \
               best_regressor.coef_[0], " ) * (" , \
               bestformula, " )")
   
    if not quiet:
        for i, yv in enumerate(y):
           print("%10.5f , %10.5f"%(yv, best_y_pred[i]))
        
    if not quiet:
        plt.scatter(best_y_pred, y,  color='black')
            
        i = 0
        for x,y in zip(best_y_pred,y):
            label = labels[i]
            
            plt.annotate(label, # this is the text
                         (x,y), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(2,2), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
            
            i += 1
            
        #plt.xticks(())
        #plt.yticks(())
            
        plt.title(sheetname + " " + str(best_regressor.coef_) + \
                    " * " + bestformula + " + " + str(best_regressor.intercept_))
            
        plt.xlabel("Predicted values " + labelname)
        plt.ylabel("Real values " + labelname)
            
        plt.show()