import math
import numpy as np
import pandas as pd
from io import StringIO

import token
import tokenize
import warnings

######################################################################
class formula_gen:

    def __init__ (self, gentype, variables):
        self.__getype__ = gentype
        self.__variables__ = variables
        self.__formulas__ = None
        self.__newfeatures__ = None

    def __get_new_feature__ (self, datain, formula):
        
        from numpy import exp, sqrt, fabs, log, power, multiply, divide

        data = pd.DataFrame(datain)
        #print(data.columns)

        sio = StringIO(formula)
        tokens = tokenize.generate_tokens(sio.readline)

        variables = []
        for toknum, tokval, _, _, _  in tokens:
            if toknum == token.NAME:
                if (tokval != "exp") and (tokval != "sqrt") \
                    and (tokval != "fabs") and (tokval != "log") \
                    and (tokval != "power") and (tokval != "multiply") \
                    and (tokval != "divide"):
                    variables.append(tokval)
                    if not (tokval in data.columns):
                        raise NameError("Error ", tokval, \
                            " not in or undefined function ")
        toexe = ""
        for vname in variables:
            toexe += vname + " = np.array(data[\""+vname+"\"].tolist(), dtype=np.float64)"
            toexe += "\n" 
            #toexe += "print(type("+vname+"), "+vname+".shape )"
            #toexe += "\n"
            #toexe += "print("+vname+")"
            #toexe += "\n"
        
        exec(toexe)
        returnvalues = []
        warnings.filterwarnings("error")
        try:
            exec("eq = " + formula)
            #exec("print(returnvalues)")
            toexe = "for value in eq:"+ \
                    "   returnvalues.append(value)"
            exec(toexe)
        except RuntimeWarning as e:
            return [None]
        
        warnings.resetwarnings()

        return returnvalues
    
    
    def __fit_gen1__ (self):

        features = self.__variables__
        #generate the formula
        formulas = []
        numer = []
        denom = []

        for classe in features.keys():
            dim = len(features[classe])

            f1 = []
            f2 = []
            f3 = []
            f4 = []
            f5 = []
            f6 = []
            f7 = []
            f8 = []

            for i in range(dim):
                f1.append(features[classe][i] )
                f2.append("power("+features[classe][i] + ", 2)")
                f3.append("power("+features[classe][i] + ", 3)")
                f4.append("power("+features[classe][i] + ", 4)")
                f5.append("power("+features[classe][i] + ", 5)")
                f6.append("exp("+features[classe][i] + ")")
                f7.append("sqrt(fabs("+features[classe][i] + "))")
                f8.append("log(fabs("+features[classe][i] + "))")

            ftuple = (f1, f2, f3, f4, f5, f6, f7, f8)

            for i in range(len(ftuple)):
                first = ftuple[i]
                for j in range(i, len(ftuple)):
                    second = ftuple[j]
                    for f in first:
                        for s in second:
                            if f != s:
                                numer.append(f + " + " + s)
                                numer.append(f + " - " + s)
                                numer.append("multiply("+ f + " , " + s + ")")

        for classe in features:
            dim = len(features[classe])
            for i in range(dim):
                denom.append("exp("+features[classe][i]+")")
                denom.append("power("+features[classe][i]+", 2)")
                denom.append("power("+features[classe][i]+", 3)")
                denom.append("power("+features[classe][i]+", 4)")
                denom.append("power("+features[classe][i]+", 5)")
                denom.append("sqrt(fabs("+features[classe][i] + "))")
                denom.append("log(fabs("+features[classe][i] + "))")
        
        for n in numer:
            for d in denom:
                formulas.append("divide(("+n+"), ("+d+"))")
            
        if len(formulas) != len(set(formulas)):
            formulas = list(set(formulas)) 
        
        return formulas
        

    def fit (self, x, y):
        self.__formulas__ = []
        if self.__getype__ == "gen1":
           self.__formulas__ = self.__fit_gen1__ ()

        data = {}
        i = 0
        for c in self.__variables__:
            for bf in self.__variables__[c]:
                if i > x.shape[1]:
                    raise Exception("Error: too many basic features") 

                data[bf] = list(x[:,i])
                i += 1

        fidxtorm = []
        self.__newfeatures__ = []
        for idxf, f in enumerate(self.__formulas__):
            newf = self.__get_new_feature__(data, f) 
            if newf[0] == None:
                fidxtorm.append(idxf)
                #print(" torm :", f)
                self.__newfeatures__.append([])
            else:
                self.__newfeatures__.append(newf)
                #print(f)
                #for i in range(len(newf)):
                #    v = data["v"][i]
                #    cE = data["cE"][i]
                #    print(v, cE)
                #    print((v+cE) / math.exp(v),  newf[i])

        for index in sorted(fidxtorm, reverse=True):
            del self.__newfeatures__[index]
            del self.__formulas__[index]
                
        return 

    def predict (self, x, verbose=0):
        pred_y = []
        for i in range(x.shape[0]):
            yval = []
            for j in range(x.shape[1]):
                yval.append(0.0)
            pred_y.append (0.0)
        return np.asarray(pred_y)

######################################################################

def build_model (gentype, variables):
    
    model = formula_gen (gentype, variables)

    return model 