import sys
import math
import time 
import token
import scipy
import parser
import tokenize

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from concurrent import futures

#from sklearn.utils import check_arrays

###############################################################################

def mean_absolute_percentage_error(y_true, y_pred):
    
    #y_true, y_pred = check_arrays(y_true, y_pred)

    v_med = np.mean(y_true)

    allv = []
    for i in range(y_true.shape[0]):
        v_pred = y_pred[i]
        v_true = y_true[i]

        v = 0.0
        if v_true == 0.0 and v_pred != 0.0:
            v = math.fabs((v_med - v_pred)/v_med)
        elif v_true == 0.0 and v_pred == 0.0:
            v = 0.0
        else:
            v = math.fabs((v_true - v_pred)/v_true)
            #print(v, v_true, v_pred)

        allv.append(v)

    return np.mean(allv) * 100

###############################################################################

from io import StringIO

def progress_bar (count, total, status=''):

    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush() 

###############################################################################

def generate_formulas (features):

    formulas  = []

    numer = []
    for classe in features:

        dim = len(features[classe])
        
        f1 = []
        f2 = []
        f3 = []
        f4 = []
        f5 = []
        for i in range(dim):
            
            f1.append(features[classe][i] )
            f2.append(features[classe][i] + "**3")
            f3.append(features[classe][i] + "**5")
            #f4.append("sqrt(fabs("+features[classe][i] + "))")
            f5.append("exp("+features[classe][i] + ")")
        
        ftuple = (f1, f2, f3, f4, f5)

        for i in range(len(ftuple)):
            first = ftuple[i]
            for j in range(i, len(ftuple)):
                second = ftuple[j]
                for f in first:
                    for s in second:
                        if f != s:
                            numer.append(f + " + " + s)
                            numer.append(f + " - " + s)

    deno = []
    for classe in features:
        dim = len(features[classe])
        for i in range(dim):
            #deno.append("sqrt(fabs("+features[classe][i]+"))")
            deno.append("exp("+features[classe][i]+")")
            deno.append("("+features[classe][i]+"**3)")
            deno.append("("+features[classe][i]+"**5)")

    for n in numer:
        for d in deno:
            formulas.append("("+n+")/("+d+")")
            
    if len(formulas) != len(set(formulas)):
        formulas = list(set(formulas)) 

    return formulas

###############################################################################

def get_twofeatures (n):

    ns = n.split(" + ")
    if len(ns) != 2:
        ns = n.split(" - ")

    fns = ns[0].replace ("**2", "")
    fns = fns.replace ("**3", "")
    fns = fns.replace ("sqrt(fabs(", "")
    fns = fns.replace ("))", "")
    fns = fns.replace ("exp(", "")
    fns = fns.replace (")", "")

    sns = ns[1].replace ("**2", "")
    sns = sns.replace ("**3", "")
    sns = sns.replace ("sqrt(fabs(", "")
    sns = sns.replace ("))", "")
    sns = sns.replace ("exp(", "")
    sns = sns.replace (")", "")

    return fns, sns

###############################################################################

def generate_formulas_four (features):

    formulas  = []

    numer = []
    deno = []
    for classe in features:

        dim = len(features[classe])
        
        f1 = []
        f2 = []
        f3 = []
        f4 = []
        f5 = []
        for i in range(dim):
            
            f1.append(features[classe][i] )
            f2.append(features[classe][i] + "**3")
            f3.append(features[classe][i] + "**5")
            #f4.append("sqrt(fabs("+features[classe][i] + "))")
            f5.append("exp("+features[classe][i] + ")")
        
        ftuple = (f1, f2, f3, f4, f5)

        for i in range(len(ftuple)):
            first = ftuple[i]
            for j in range(i, len(ftuple)):
                second = ftuple[j]
                for f in first:
                    for s in second:
                        if f != s:
                            numer.append(f + " + " + s)
                            numer.append(f + " - " + s)
                            deno.append(f + " + " + s)
                            deno.append(f + " - " + s)


    for n in numer:
        for d in deno:
            if n != d:
                bfset = set()
                fn, sn = get_twofeatures (n)
                fd, sd = get_twofeatures (d)
                bfset.add(fn)
                bfset.add(sn)
                bfset.add(fd)
                bfset.add(sd)

                if len(bfset) == 4:
                    #print(bfset)
                    formulas.append("("+n+")/("+d+")")
            
    if len(formulas) != len(set(formulas)):
        formulas = list(set(formulas)) 

    return formulas

###############################################################################

def generate_formulas_AB (features, atomicdata, lista, listb, method = 1):

    formulas = []
    newdataframeAB = {}
    
    if lista.shape != listb.shape:
        raise NameError("Error lists shape error ")
    
    newdataframeAB["Name"] = []
    for k in atomicdata.to_dict():
        newdataframeAB[k+"_A"] = []
        newdataframeAB[k+"_B"] = []
    
    for i in range(lista.shape[0]):
        
        aelmnt = atomicdata.index[atomicdata['Z'] == \
                                  lista[i]].tolist()[0]
        belmnt = atomicdata.index[atomicdata['Z'] == \
                                  listb[i]].tolist()[0]
                                  
        newdataframeAB["Name"].append(str(aelmnt)+ \
                                      "_"+str(belmnt))
                                  
        adict = atomicdata.loc[aelmnt, :].to_dict()
        bdict = atomicdata.loc[belmnt, :].to_dict()
        
        for k in atomicdata.to_dict():
            newdataframeAB[k+"_A"].append(adict[k])
            newdataframeAB[k+"_B"].append(bdict[k])
    
    featuresAB = pd.DataFrame.from_dict(newdataframeAB) 
     
    if method == 1:

        numer = []
        deno = []
    
        for classe in features:
            dim = len(features[classe])
            
            f1 = []
            f2 = []
            f3 = []
            f4 = []
            f5 = []
            for i in range(dim):
                
                f1.append(features[classe][i] + "_A")
                f1.append(features[classe][i] + "_B")
                
                f2.append(features[classe][i] + "_A**2")
                f2.append(features[classe][i] + "_B**2")
                
                f3.append(features[classe][i] + "_A**3")
                f3.append(features[classe][i] + "_B**3")
                
                f4.append("sqrt(fabs("+features[classe][i] + "_A))")
                f4.append("sqrt(fabs("+features[classe][i] + "_B))")
                
                f5.append("exp("+features[classe][i] + "_A)")
                f5.append("exp("+features[classe][i] + "_B)")
    
            
            ftuple = (f1, f2, f3, f4, f5)
            
            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i, len(ftuple)):
                    second = ftuple[j]
                    for f in first:
                        for s in second:
                            if f != s:
                                numer.append(f + " + " + s)
                                numer.append(f + " - " + s)
    
        for classe in features:
            dim = len(features[classe])
            for i in range(dim):
                deno.append("sqrt(fabs("+features[classe][i]+"_A))")
                deno.append("exp("+features[classe][i]+"_A)")
                deno.append("("+features[classe][i]+"_A**2)")
                deno.append("("+features[classe][i]+"_A**3)")
                deno.append("sqrt(fabs("+features[classe][i]+"_B))")
                deno.append("exp("+features[classe][i]+"_B)")
                deno.append("("+features[classe][i]+"_B**2)")
                deno.append("("+features[classe][i]+"_B**3)")

        for n in numer:
            for d in deno:
                formulas.append("("+n+")/("+d+")")
                
    elif method == 2:

        numer = []
        numertype = []

        deno = []
        denotype = []
        
        for classe in features:
            dim = len(features[classe])
            
            f1 = []
            f2 = []
            f3 = []
            f4 = []
            f5 = []
            for i in range(dim):
                
                f1.append(features[classe][i] + "_A")
                f1.append(features[classe][i] + "_B")
                
                f2.append(features[classe][i] + "_A**2")
                f2.append(features[classe][i] + "_B**2")
                
                f3.append(features[classe][i] + "_A**3")
                f3.append(features[classe][i] + "_B**3")
                
                f4.append("sqrt(fabs("+features[classe][i] + "_A))")
                f4.append("sqrt(fabs("+features[classe][i] + "_B))")
                
                f5.append("exp("+features[classe][i] + "_A)")
                f5.append("exp("+features[classe][i] + "_B)")
    
            
            ftuple = (f1, f2, f3, f4, f5)
            
            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i, len(ftuple)):
                    second = ftuple[j]
                    for f in first:
                        for s in second:
                            if f != s:
                                numer.append(f + " + " + s)
                                numertype.append(classe)
                                numer.append(f + " - " + s)
                                numertype.append(classe)
    
        for classe in features:
            dim = len(features[classe])
            for i in range(dim):
                deno.append("sqrt(fabs("+features[classe][i]+"_A))")
                denotype.append(classe)
                deno.append("exp("+features[classe][i]+"_A)")
                denotype.append(classe)
                deno.append("("+features[classe][i]+"_A**2)")
                denotype.append(classe)
                deno.append("("+features[classe][i]+"_A**3)")
                denotype.append(classe)
                deno.append("sqrt(fabs("+features[classe][i]+"_B))")
                denotype.append(classe)
                deno.append("exp("+features[classe][i]+"_B)")
                denotype.append(classe)
                deno.append("("+features[classe][i]+"_B**2)")
                denotype.append(classe)
                deno.append("("+features[classe][i]+"_B**3)")
                denotype.append(classe)
 

        for idxn, n in enumerate(numer):
            for idxd, d in enumerate(deno):
                if numertype[idxn] != denotype[idxd]:
                    formulas.append("("+n+")/("+d+")")
    
    elif method == 3:

        numer = []
        numertype = []
        
        for classe in features:
            dim = len(features[classe])

            f1 = []
            f2 = []
            f3 = []
            f4 = []
            f5 = []
            for i in range(dim):
                
                f1.append(features[classe][i] + "_A")
                f1.append(features[classe][i] + "_B")
                
                f2.append(features[classe][i] + "_A**2")
                f2.append(features[classe][i] + "_B**2")
                
                f3.append(features[classe][i] + "_A**3")
                f3.append(features[classe][i] + "_B**3")
                
                f4.append("sqrt(fabs("+features[classe][i] + "_A))")
                f4.append("sqrt(fabs("+features[classe][i] + "_B))")
                
                f5.append("exp("+features[classe][i] + "_A)")
                f5.append("exp("+features[classe][i] + "_B)")
    
            
            ftuple = (f1, f2, f3, f4, f5)

            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i, len(ftuple)):
                    second = ftuple[j]
                    for f in first:
                        for s in second:
                            if f != s:
                                numer.append(f + " + " + s)
                                numertype.append(classe)
                                numer.append(f + " - " + s)
                                numertype.append(classe)

        for idxn in range(len(numer)):
            for idxd in range(idxn+1, len(numer)):
                if (numertype[idxn] != numertype[idxd]) and (numer[idxn] != numer[idxd]) :
                    formulas.append("("+numer[idxn]+")/("+numer[idxd]+")")
 
    elif method == 4:

        numer = []
        numertype = []
        
        for classe in features:
            dim = len(features[classe])
            
            f1 = []
            f2 = []
            f3 = []
            f4 = []
            f5 = []
            for i in range(dim):
                
                f1.append(features[classe][i] + "_A")
                f1.append(features[classe][i] + "_B")
                
                f2.append(features[classe][i] + "_A**2")
                f2.append(features[classe][i] + "_B**2")
                
                f3.append(features[classe][i] + "_A**3")
                f3.append(features[classe][i] + "_B**3")
                
                f4.append("sqrt(fabs("+features[classe][i] + "_A))")
                f4.append("sqrt(fabs("+features[classe][i] + "_B))")
                
                f5.append("exp("+features[classe][i] + "_A)")
                f5.append("exp("+features[classe][i] + "_B)")
    
            
            ftuple = (f1, f2, f3, f4, f5)
            
            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i, len(ftuple)):
                    second = ftuple[j]
                    for f in first:
                        for s in second:
                            if f != s:

                                if (i == 0 and j == 0)  or \
                                    (i == 1 and j == 1)  or \
                                    (i == 2 and j == 2):                                        
                                    numer.append(f + " + " + s)
                                    numertype.append(classe)
                                    numer.append(f + " - " + s)
                                    numertype.append(classe)
                                if (i == 3 and j == 3)  or \
                                    (i == 4 and j == 4):
                                    numer.append(f + " * " + s)
                                    numer.append(f + " / " + s)
                                    numer.append(s + " / " + f)
                                    numertype.append(classe)
                                if (i == 0 and j == 3)  or \
                                    (i == 1 and j == 3)  or \
                                    (i == 2 and j == 3):
                                    numer.append(f + " * " + s)
                                    numertype.append(classe)

        for idxn in range(len(numer)):
            for idxd in range(idxn+1, len(numer)):
                formulas.append("("+numer[idxn]+")/("+numer[idxd]+")")
            
    if len(formulas) != len(set(formulas)):
        formulas = list(set(formulas)) 

    return formulas, featuresAB

###############################################################################

def generate_formulas_AB_mixed (features, data, method = 1):

    formulas = []
    newdataframeAB = {}

    # TODO

    for k in features:
        for f in features[k]:
            newdataframeAB[f] = list(data[f].values)

    featuresAB = pd.DataFrame.from_dict(newdataframeAB) 

    if method == 1:

        numer = []
        deno = []
    
        for classe in features:
            dim = len(features[classe])
            
            f1 = []
            f2 = []
            f3 = []
            f4 = []
            f5 = []
            for i in range(dim):
                
                f1.append(features[classe][i])
                f2.append(features[classe][i] + "**2")
                f3.append(features[classe][i] + "**3")
                f4.append("sqrt(fabs("+features[classe][i] + "))")
                f5.append("exp("+features[classe][i] + ")")
            
            ftuple = (f1, f2, f3, f4, f5)
            
            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i, len(ftuple)):
                    second = ftuple[j]
                    for f in first:
                        for s in second:
                            if f != s:
                                numer.append(f + " + " + s)
                                numer.append(f + " - " + s)
    
        for classe in features:
            dim = len(features[classe])
            for i in range(dim):
                deno.append("sqrt(fabs("+features[classe][i]+"))")
                deno.append("exp("+features[classe][i]+")")
                deno.append("("+features[classe][i]+"**2)")
                deno.append("("+features[classe][i]+"**3)")

        for n in numer:
            for d in deno:
                formulas.append("("+n+")/("+d+")")

    return formulas, featuresAB

###############################################################################

def generate_formulas_ABC (features, atomicdata, lista, listb, listc, \
    method = 1):

    formulas = []
    newdataframeABC = {}
    
    if lista.shape != listb.shape:
        raise NameError("Error lists shape error ")

    if lista.shape != listc.shape:
        raise NameError("Error lists shape error ")
    
    newdataframeABC["Name"] = []
    for k in atomicdata.to_dict():
        newdataframeABC[k+"_A"] = []
        newdataframeABC[k+"_B"] = []
        newdataframeABC[k+"_C"] = []

    labelofatnm = "A"    
    for i in range(lista.shape[0]):
        print("Looking for ", lista[i], listb[i], listc[i])

        if lista[i] not in atomicdata[labelofatnm].values:
            print("Cannot find \""+lista[i]+"\" in the list" )
                
        if listb[i] not in atomicdata[labelofatnm].values:
            print("Cannot find \""+listb[i]+"\" in the list" )
        
        if listc[i] not in atomicdata[labelofatnm].values:
            print("Cannot find \""+listc[i]+"\" in the list" )
                
        aelmnt = atomicdata.index[atomicdata[labelofatnm] == \
                                  lista[i]].tolist()[0]
        belmnt = atomicdata.index[atomicdata[labelofatnm] == \
                                  listb[i]].tolist()[0]
        celmnt = atomicdata.index[atomicdata[labelofatnm] == \
                                  listc[i]].tolist()[0]
                                        
        newdataframeABC["Name"].append(str(aelmnt)+ \
                                      "_"+str(belmnt) + \
                                      "_"+str(celmnt))
                                  
        adict = atomicdata.loc[aelmnt, :].to_dict()
        bdict = atomicdata.loc[belmnt, :].to_dict()
        cdict = atomicdata.loc[celmnt, :].to_dict()
        
        for k in atomicdata.to_dict():
            newdataframeABC[k+"_A"].append(adict[k])
            newdataframeABC[k+"_B"].append(bdict[k])
            newdataframeABC[k+"_C"].append(cdict[k])
    
    featuresABC = pd.DataFrame.from_dict(newdataframeABC) 
    
    numer = []
    deno = []

    if method == 1 or method == 3:
        for classe in features:
            dim = len(features[classe])
        
            f1 = []
            f2 = []
            f3 = []
            f4 = []
            f5 = []
            for i in range(dim):
            
                f1.append(features[classe][i] + "_A")
                f1.append(features[classe][i] + "_B")
                f1.append(features[classe][i] + "_C")

                f2.append(features[classe][i] + "_A**2")
                f2.append(features[classe][i] + "_B**2")
                f2.append(features[classe][i] + "_C**2")
            
                f3.append(features[classe][i] + "_A**3")
                f3.append(features[classe][i] + "_B**3")
                f3.append(features[classe][i] + "_C**3")
            
                f4.append("sqrt(fabs("+features[classe][i] + "_A))")
                f4.append("sqrt(fabs("+features[classe][i] + "_B))")
                f4.append("sqrt(fabs("+features[classe][i] + "_C))")
            
                f5.append("exp("+features[classe][i] + "_A)")
                f5.append("exp("+features[classe][i] + "_B)")
                f5.append("exp("+features[classe][i] + "_C)")
        
            ftuple = (f1, f2, f3, f4, f5)
        
            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i+1, len(ftuple)):
                    second = ftuple[j]
                    for f in first:
                        for s in second:
                            if f != s:
                                numer.append(f + " + " + s)
                                numer.append(f + " - " + s)

        for classe in features:
            dim = len(features[classe])
            for i in range(dim):
                deno.append("sqrt(fabs("+features[classe][i]+"_A))")
                deno.append("exp("+features[classe][i]+"_A)")
                deno.append("("+features[classe][i]+"_A**2)")
                deno.append("("+features[classe][i]+"_A**3)")
                deno.append("sqrt(fabs("+features[classe][i]+"_B))")
                deno.append("exp("+features[classe][i]+"_B)")
                deno.append("("+features[classe][i]+"_B**2)")
                deno.append("("+features[classe][i]+"_B**3)")
                deno.append("sqrt(fabs("+features[classe][i]+"_C))")
                deno.append("exp("+features[classe][i]+"_C)")
                deno.append("("+features[classe][i]+"_C**2)")
                deno.append("("+features[classe][i]+"_C**3)")

    elif method == 2:
        for classe in features:
            dim = len(features[classe])
        
            f1 = []
            f2 = []
            f3 = []
            f4 = []
            f5 = []
            for i in range(dim):
            
                f1.append(features[classe][i] + "_A")
                f1.append(features[classe][i] + "_B")
                f1.append(features[classe][i] + "_C")

                f2.append(features[classe][i] + "_A**2")
                f2.append(features[classe][i] + "_B**2")
                f2.append(features[classe][i] + "_C**2")
            
                f3.append(features[classe][i] + "_A**3")
                f3.append(features[classe][i] + "_B**3")
                f3.append(features[classe][i] + "_C**3")
            
                f4.append("sqrt(fabs("+features[classe][i] + "_A))")
                f4.append("sqrt(fabs("+features[classe][i] + "_B))")
                f4.append("sqrt(fabs("+features[classe][i] + "_C))")
            
                f5.append("exp("+features[classe][i] + "_A)")
                f5.append("exp("+features[classe][i] + "_B)")
                f5.append("exp("+features[classe][i] + "_C)")

        
            ftuple = (f1, f2, f3, f4, f5)
        
            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i+1, len(ftuple)):
                    second = ftuple[j]
                    for f in first:
                        for s in second:
                            if f != s:
                                numer.append(f + " + " + s)
                                numer.append(f + " - " + s)
                                deno.append(f + " + " + s)
                                deno.append(f + " + " + s)

    elif method == 4: # not used 
        for classe in features:
            dim = len(features[classe])
        
            f1 = []
            f2 = []
            f3 = []
            for i in range(dim):
            
                f1.append(features[classe][i] + "_A")
                f1.append(features[classe][i] + "_A**2")
                f1.append(features[classe][i] + "_A**3")
                f1.append("sqrt(fabs("+features[classe][i] + "_A))")
                f1.append("exp("+features[classe][i] + "_A)")

                f2.append(features[classe][i] + "_B")
                f2.append(features[classe][i] + "_B**2")
                f2.append(features[classe][i] + "_B**3")
                f2.append("sqrt(fabs("+features[classe][i] + "_B))")
                f2.append("exp("+features[classe][i] + "_B)")
            
                f3.append(features[classe][i] + "_C")
                f3.append(features[classe][i] + "_C**2")
                f3.append(features[classe][i] + "_C**3")
                f3.append("sqrt(fabs("+features[classe][i] + "_C))")
                f3.append("exp("+features[classe][i] + "_C)")

        
            ftuple = (f1, f2, f3)
        
            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i+1, len(ftuple)):
                    second = ftuple[j]
                    for k in range(j+1, len(ftuple)):
                        third = ftuple[k]
                        for f in first:
                            for s in second:
                                for t in third:
                                    if f != s and t != f and t != s:
                                        numer.append(f + " + " + s)
                                        numer.append(f + " - " + s)
                                        numer.append(f + " + " + t)
                                        numer.append(f + " - " + t)
                                        numer.append(s + " + " + t)
                                        numer.append(s + " - " + t)
                                        deno.append(f)
                                        deno.append(s)
                                        deno.append(t)


    print("Deal with ", len(numer), " numer and ", len(deno), " denom")                                   
    for n in numer:
        for d in deno:
            if d != n:
                if method == 3 or method == 4:

                    atomfound = 0
                    for tof in ["_A", "_B", "_C"]:
                        if d.find(tof) >= 0:
                            atomfound += 1
                        else:
                            if n.find(tof) >= 0:
                                atomfound += 1
                    
                    if atomfound == 3:
                        formulas.append("("+n+")/("+d+")")
                else:
                    formulas.append("("+n+")/("+d+")")
            
    if len(formulas) != len(set(formulas)):
        formulas = list(set(formulas)) 

    return formulas, featuresABC

###############################################################################

def get_new_feature (indatframe, formula):

    from math import exp, sqrt, fabs

    code = parser.expr(formula).compile()
    sio = StringIO(formula)
    tokens = tokenize.generate_tokens(sio.readline)
    
    variables = []
    for toknum, tokval, _, _, _  in tokens:
        if toknum == token.NAME:
            if (tokval != "exp") and (tokval != "sqrt") and (tokval != "fabs"):
                variables.append(tokval)
                if not (tokval in indatframe.columns):
                    raise NameError("Error ", tokval, \
                            " not in or undefined function ")

    todefinevars = ""
    for vname in variables:
        exec(vname + "_list = []")
        todefinevars += vname + " = None\n"
        for v in indatframe[vname].tolist():
            exec(vname + "_list.append("+str(v)+")")

    returnvalues = []
    # define the needed constant 
    exec(todefinevars)
    for i in range(len(indatframe[variables[0]].tolist())):
        for vname in variables:
            exec(vname + " = " + vname + "_list["+str(i)+"]")
        
        nf = eval(code)
        returnvalues.append(nf)

    return  returnvalues

###############################################################################

def feature_check_fullsetrmse(feature_list_indexes, dataset_features, y_array):

    fd = dict()

    fd['formulas'] = []
    fd['index']    = []
    fd['rmse']     = []
    fd['percoeff'] = []
    fd['pval']     = []
    fd['mape']     = []
    fd['r2']       = []
    
    dataset_keys = dataset_features.keys()[feature_list_indexes]
    for jj,keyv in enumerate(dataset_keys):
        X = dataset_features[keyv]

        val1, val2 = \
                scipy.stats.pearsonr(dataset_features[keyv].values, \
                y_array.reshape(dataset_features[keyv].values.shape[0]))
        
        regressor = LinearRegression()
        regressor.fit((np.array(X)).reshape(-1,1), y_array)
        y_pred = regressor.predict((np.array(X)).reshape(-1,1))

        r2val = r2_score(y_array, y_pred)

        mse = mean_squared_error(y_array, y_pred)
        rmse = math.sqrt(mse)
        mape = mean_absolute_percentage_error(y_array, y_pred)

        progress_bar(jj + 1, len(dataset_keys))

        fd['formulas'].append(keyv)
        fd['index'].append(jj)
        fd['rmse'].append(rmse)
        fd['mape'].append(np.average(mape))
        fd['r2'].append(r2val)

        if (math.isnan(val1)):
            fd['percoeff'].append(np.fabs(0.0))
        else:
            fd['percoeff'].append(np.fabs(val1))

        fd['pval'].append(val2)

    feature_mse_dataframe = pd.DataFrame.from_dict(fd)
    fd2 = feature_mse_dataframe.copy()
    
    print("")
    
    return feature_mse_dataframe

###############################################################################

def feature_check_lr(feature_list_indexes, dataset_features, y_array, \
        numoflr = 1000):

    fd = dict()

    fd['formulas'] = []
    fd['index']    = []
    fd['mse']      = []
    fd['percoeff'] = []
    fd['pval']     = []
    fd['mape']     = []
    fd['r2']       = []
    
    dataset_keys = dataset_features.keys()[feature_list_indexes]
    for jj,keyv in enumerate(dataset_keys):
        X = dataset_features[keyv]

        val1, val2 = \
                scipy.stats.pearsonr(dataset_features[keyv].values, \
                y_array.reshape(dataset_features[keyv].values.shape[0]))
        
        mse = []
        mape = []
        r2 = []
        for ii in range(numoflr):
            X_train, X_test, y_train, y_test = \
                    train_test_split(X, y_array, test_size=0.1, random_state=ii)
            regressor = LinearRegression()
            regressor.fit((np.array(X_train)).reshape(-1,1), y_train)
            y_pred = regressor.predict((np.array(X_test)).reshape(-1,1))

            r2.append(r2_score(y_test,y_pred))
            mse.append(mean_squared_error(y_test,y_pred))
            mape.append(mean_absolute_percentage_error(y_test, y_pred))

        avg = float(np.average(mse))

        progress_bar(jj + 1, len(dataset_keys))
        fd['formulas'].append(keyv)
        fd['index'].append(jj)
        fd['mse'].append(avg)
        fd['mape'].append(np.average(mape))

        if (math.isnan(val1)):
            fd['percoeff'].append(np.fabs(0.0))
        else:
            fd['percoeff'].append(np.fabs(val1))

        fd['pval'].append(val2)
        fd['r2'].append(np.average(r2))

    feature_mse_dataframe = pd.DataFrame.from_dict(fd)
    fd2 = feature_mse_dataframe.copy()
    
    print("")
    
    return feature_mse_dataframe

###############################################################################

def task_feature2D_check_lr (inps):

    f1 = inps[0]
    f2 = inps[1]
    dataset_features = inps[2]
    numoflr = inps[3]
    y_array = inps[4]
    toprint = inps[5]

    avg = float("inf")
    if f1 in dataset_features.columns and \
            f2 in dataset_features.columns:
        Xdf = dataset_features[[f1, f2]].copy()
        X = Xdf.values
    
        mse = []
        for ii in range(numoflr):
            X_train, X_test, y_train, y_test = \
                    train_test_split(X, y_array, test_size=0.1, random_state=ii)
            regressor = LinearRegression()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            mse.append(mean_squared_error(y_test,y_pred))
    
        avg = float(np.average(mse))

    if toprint != "":
        print(toprint)

    return (f1, f2, avg)

###############################################################################

def task_feature2D_check_lr_msefullset (inps):

    f1 = inps[0]
    f2 = inps[1]
    dataset_features = inps[2]
    numoflr = inps[3]
    y_array = inps[4]
    toprint = inps[5]

    mse = float("inf")
    if f1 in dataset_features.columns and \
            f2 in dataset_features.columns:
        Xdf = dataset_features[[f1, f2]].copy()
        X = Xdf.values
    
        regressor = LinearRegression()
        regressor.fit(X, y_array)
        y_pred = regressor.predict(X)
        mse = mean_squared_error(y_array,y_pred)
    
    if toprint != "":
        print(toprint)

    return (f1, f2, mse)

###############################################################################

def feature2D_check_lr(twoDformulas, dataset_features, y_array, nt, \
        numoflr = 1000, showiter=False, goodfiles=[]):

    fd = dict()

    fd['formulas'] = []
    fd['mse']     = []

    if nt == 1:

        idx = 0
        dim = len(twoDformulas)
        avgtime = 0.0
        for f1, f2 in twoDformulas:
            start = time.time()

            Xdf = dataset_features[[f1, f2]].copy()
            X = Xdf.values
        
            mse = []
            for ii in range(numoflr):
                X_train, X_test, y_train, y_test = \
                        train_test_split(X, y_array, test_size=0.1, random_state=ii)
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                mse.append(mean_squared_error(y_test,y_pred))
        
            avg = float(np.average(mse))

            fd['formulas'].append((f1, f2))
            fd['mse'].append(avg)

            end = time.time()

            idx += 1
            if showiter:
                avgtime += (end - start)
                est = (float(dim)*(avgtime/float(idx)))/3600.0
                fname = ""
                if len(goodfiles) != 0:
                    fname = goodfiles[idx-1]
                print("Iter %10d of %10d [%10.6f estimated tot. %10.6f hrs.] %s"%(idx, dim, \
                        (end - start), est, fname),flush=True)
            else:
                progress_bar(idx, dim)
    else:

        print("Preparing input for ", nt, " threads/processes")
        inputs = []
        toprint = ""
        idx = 0
        dim = len(twoDformulas)
        for f1, f2 in twoDformulas:
            inputs.append((f1, f2, dataset_features,
                    numoflr, y_array, toprint))
            idx += 1
            if idx == int(dim*0.10) or \
                idx == int(dim*0.20) or \
                idx == int(dim*0.30) or \
                idx == int(dim*0.40) or \
                idx == int(dim*0.50) or \
                idx == int(dim*0.60) or \
                idx == int(dim*0.70) or \
                idx == int(dim*0.80) or \
                idx == int(dim*0.90) or \
                idx == int(dim*0.99):
                    toprint = "%15d of %15d"%(idx, dim) 
            else:
                toprint = ""
        
        #with futures.ThreadPoolExecutor(max_workers=nt) as executor:
        with futures.ProcessPoolExecutor(max_workers=nt) as executor:
            results = executor.map(task_feature2D_check_lr, inputs)
        
        for res in list(results):
            fd['formulas'].append((res[0], res[1]))
            fd['mse'].append(res[2])

    return pd.DataFrame.from_dict(fd)

###############################################################################

def feature2D_check_lr_msefullset(twoDformulas, dataset_features, y_array, nt, \
        numoflr = 1000, showiter=False, goodfiles=[]):

    fd = dict()

    fd['formulas'] = []
    fd['mse']     = []

    if nt == 1:

        idx = 0
        dim = len(twoDformulas)
        avgtime = 0.0
        for f1, f2 in twoDformulas:
            start = time.time()

            Xdf = dataset_features[[f1, f2]].copy()
            X = Xdf.values
        
            regressor = LinearRegression()
            regressor.fit(X, y_array)
            y_pred = regressor.predict(X)
            mse = mean_squared_error(y_array,y_pred)
        
            fd['formulas'].append((f1, f2))
            fd['mse'].append(mse)

            end = time.time()

            idx += 1
            if showiter:
                avgtime += (end - start)
                est = (float(dim)*(avgtime/float(idx)))/3600.0
                fname = ""
                if len(goodfiles) != 0:
                    fname = goodfiles[idx-1]
                print("Iter %10d of %10d [%10.6f estimated tot. %10.6f hrs.] %s"%(idx, dim, \
                        (end - start), est, fname),flush=True)
            else:
                progress_bar(idx, dim)
    else:

        print("Preparing input for ", nt, " threads/processes")
        inputs = []
        toprint = ""
        idx = 0
        dim = len(twoDformulas)
        for f1, f2 in twoDformulas:
            inputs.append((f1, f2, dataset_features,
                    numoflr, y_array, toprint))
            idx += 1
            if idx == int(dim*0.10) or \
                idx == int(dim*0.20) or \
                idx == int(dim*0.30) or \
                idx == int(dim*0.40) or \
                idx == int(dim*0.50) or \
                idx == int(dim*0.60) or \
                idx == int(dim*0.70) or \
                idx == int(dim*0.80) or \
                idx == int(dim*0.90) or \
                idx == int(dim*0.99):
                    toprint = "%15d of %15d"%(idx, dim) 
            else:
                toprint = ""
        
        #with futures.ThreadPoolExecutor(max_workers=nt) as executor:
        with futures.ProcessPoolExecutor(max_workers=nt) as executor:
            results = executor.map(task_feature2D_check_lr_msefullset, inputs)
        
        for res in list(results):
            fd['formulas'].append((res[0], res[1]))
            fd['mse'].append(res[2])

    return pd.DataFrame.from_dict(fd)

###############################################################################

def checkcorr (inp):
    
    start1dfeatures = inp[0]
    idx1 = inp[1]
    f1 = inp[2]

    res = {}
    
    idx2 = 0
    for f2 in start1dfeatures["formulas"]:
        if idx2 > idx1:
            if f1 != f2:
                Xdf = featuresvalue[[f1, f2]].copy()
                # check correlation
                corrval = np.fabs(Xdf.corr().values[0,1])
                res[(f1 , f2)] = corrval
        idx2 += 1

    dim = len(start1dfeatures["formulas"])
    if idx1 == int(dim*0.10) or \
            idx1 == int(dim*0.20) or \
            idx1 == int(dim*0.30) or \
            idx1 == int(dim*0.40) or \
            idx1 == int(dim*0.50) or \
            idx1 == int(dim*0.60) or \
            idx1 == int(dim*0.70) or \
            idx1 == int(dim*0.80) or \
            idx1 == int(dim*0.90) or \
            idx1 == int(dim*0.99):
        print( "%15d of %15d"%(idx1, dim) )

    return res

###############################################################################

def feature3D_check_lr(threeDformulas, dataset_features, y_array, \
        numoflr = 1000, showiter=False, goodfiles=[]):

    fd = dict()

    fd['formulas'] = []
    fd['mse']     = []

    idx = 0
    dim = len(threeDformulas)
    avgtime = 0.0
    for f1, f2, f3 in threeDformulas:
        start = time.time()

        Xdf = None
        try:
            Xdf = dataset_features[[f1, f2, f3]].copy()

            #print("\""+f1+"\"\n\""+f2+"\"\n\""+f3+"\"\nKey Found") 

            X = Xdf.values
    
            mse = []
            for ii in range(numoflr):
                X_train, X_test, y_train, y_test = \
                        train_test_split(X, y_array, test_size=0.1, random_state=ii)
                regressor = LinearRegression()
                regressor.fit(X_train, y_train)
                y_pred = regressor.predict(X_test)
                mse.append(mean_squared_error(y_test,y_pred))
            
            avg = float(np.average(mse))
            
            fd['formulas'].append((f1, f2, f3))
            fd['mse'].append(avg)
            
            end = time.time()
            
            if showiter:
                avgtime += (end - start)
                est = (float(dim)*(avgtime/float(idx+1)))/3600.0
                fname = ""
                if len(goodfiles) != 0:
                    fname = goodfiles[idx]
                print("Iter %10d of %10d [%10.6f estimated tot. %10.6f hrs.] %s"%(idx+1, dim, \
                        (end - start), est, fname),flush=True)
            else:
                progress_bar(idx, dim)

        except KeyError as ki:
            print("\""+f1+"\"\n\""+f2+"\"\n\""+f3+"\"\nKey Error") 
        
        idx += 1


    return pd.DataFrame.from_dict(fd)

###############################################################################

def feature3D_check_lr_msefullset(threeDformulas, dataset_features, y_array, \
        numoflr = 1000, showiter=False, goodfiles=[]):

    fd = dict()

    fd['formulas'] = []
    fd['mse']     = []

    idx = 0
    dim = len(threeDformulas)
    avgtime = 0.0
    for f1, f2, f3 in threeDformulas:
        start = time.time()

        Xdf = None
        try:
            Xdf = dataset_features[[f1, f2, f3]].copy()

            #print("\""+f1+"\"\n\""+f2+"\"\n\""+f3+"\"\nKey Found") 

            X = Xdf.values
    
            regressor = LinearRegression()
            regressor.fit(X, y_array)
            y_pred = regressor.predict(X)
            mse = (mean_squared_error(y_array, y_pred))
            
            fd['formulas'].append((f1, f2, f3))
            fd['mse'].append(mse)
            
            end = time.time()
            
            if showiter:
                avgtime += (end - start)
                est = (float(dim)*(avgtime/float(idx+1)))/3600.0
                fname = ""
                if len(goodfiles) != 0:
                    fname = goodfiles[idx]
                print("Iter %10d of %10d [%10.6f estimated tot. %10.6f hrs.] %s"%(idx+1, dim, \
                        (end - start), est, fname),flush=True)
            else:
                progress_bar(idx, dim)

        except KeyError as ki:
            print("\""+f1+"\"\n\""+f2+"\"\n\""+f3+"\"\nKey Error") 
        
        idx += 1

    return pd.DataFrame.from_dict(fd)

