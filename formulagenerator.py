import numpy as np
import pandas as pd
from io import StringIO

import token
import tokenize

######################################################################

class formula_gen:

    def __init__ (self, gentype, variables):
        self.__getype__ = gentype
        self.__variables__ = variables

    def __get_new_feature__ (self, datain, formula):
        
        from numpy import exp, sqrt, fabs, log, power, multiply, divide

        data = pd.DataFrame(datain)

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
                    
        todefinevars = ""
        for vname in variables:
            exec(vname + "_list = []")
            todefinevars += vname + " = None\n"
            for v in data[vname].tolist():
                exec(vname + "_list.append("+str(v)+")")  

        returnvalues = []
        exec(todefinevars)
        for i in range(len(data[variables[0]].tolist())):
            for vname in variables:
                exec(vname + " = " + vname + "_list["+str(i)+"]")
            try:
                nf = eval(code)
            except:
                return [None]
            
            returnvalues.append(nf)

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
                f2.append(features[classe][i] + "**2")
                f3.append(features[classe][i] + "**3")
                f4.append(features[classe][i] + "**4")
                f5.append(features[classe][i] + "**5")
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
                                numer.append(f + " * " + s)

        for classe in features:
            dim = len(features[classe])
            for i in range(dim):
                denom.append("exp("+features[classe][i]+")")
                denom.append("("+features[classe][i]+"**3)")
                denom.append("("+features[classe][i]+"**5)")
                denom.append("sqrt(fabs("+features[classe][i] + "))")
                denom.append(features[classe][i] + "**2")
                denom.append(features[classe][i] + "**4")
                denom.append("log(fabs("+features[classe][i] + "))")
        
        for n in numer:
            for d in denom:
                formulas.append("("+n+")/("+d+")")
            
        if len(formulas) != len(set(formulas)):
            formulas = list(set(formulas)) 
        
        return formulas
        

    def fit (self, x, y):
        formulas = []
        if self.__getype__ == "gen1":
           formulas = self.__fit_gen1__ ()

        data = {}
        i = 0
        for c in self.__variables__:
            for bf in self.__variables__[c]:
                if i > x.shape[1]:
                    raise Exception("Error: too many basic features") 

                data[bf] = list(x[:,i])
                i += 1

        fidxtorm = []
        for idxf, f in enumerate(formulas):
            newf = self.__get_new_feature__(data, f) 
            if newf[0] == None:
                fidxtorm.append(idxf)
                #print(" torm :", f)
            else:
                print(f)

                for i in range(len(newf)):
                    v = data["v"][i]
                    cE = data["cE"][i]
                    print(v)
                    print(cE)
                    print(eval(f))
                    print(newf[i])
                exit()

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