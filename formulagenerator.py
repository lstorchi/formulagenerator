import math
import numpy as np
import pandas as pd
from io import StringIO

import token
import tokenize
import warnings

from sklearn.linear_model import LinearRegression

######################################################################
class formula_gen:

    def __init__ (self, gentype, variables):
        self.__getype__ = gentype
        self.__variables__ = variables
        self.__formulas__ = None

        self.__newfeatures__ = None
        self.__bestformula__ = None
        self.__bestlr__ = None
        self.__bestnewf__ = None


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

        return np.array(returnvalues)
    
    
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
                f2.append("("+features[classe][i] + "**2)")
                f3.append("("+features[classe][i] + "**3)")
                f4.append("("+features[classe][i] + "**4)")
                f5.append("("+features[classe][i] + "**5)")
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
                denom.append("("+features[classe][i]+"**2)")
                denom.append("("+features[classe][i]+"**3)")
                denom.append("("+features[classe][i]+"**4)")
                denom.append("("+features[classe][i]+"**5)")
                denom.append("sqrt(fabs("+features[classe][i] + "))")
                denom.append("log(fabs("+features[classe][i] + "))")
        
        for n in numer:
            for d in denom:
                formulas.append("("+n+ ") / ("+d+")")
            
        if len(formulas) != len(set(formulas)):
            formulas = list(set(formulas)) 
        
        return formulas
    
    def __refine_gen1__ (self):

        if self.__bestformula__ == None:
            raise Exception("Error: model not fitted")
        
        if self.__newfeatures__ == None:
            raise Exception("Error: new features not generated")

        raise Exception("Error: not implemented")


    def fit (self, x, y):

        if type(x) != np.ndarray:
            raise Exception("Error: x should be a numpy array")

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

        bestmse = float("inf") 
        for i in range(len(self.__newfeatures__)):
            lr = LinearRegression()
            #print("formula: ", self.__formulas__[i])
            lr.fit(self.__newfeatures__[i].reshape(-1, 1), y)
            #print("coef: ", lr.coef_)
            #print("intercept: ", lr.intercept_)
            #print("score: ", lr.score(self.__newfeatures__[i].reshape(-1, 1), y))
            mse = np.mean((lr.predict(self.__newfeatures__[i].reshape(-1, 1)) - y) ** 2)
            if mse < bestmse:
                bestmse = mse
                self.__bestformula__ = self.__formulas__[i]
                self.__bestlr__ = lr
                self.__bestnewf__ = self.__newfeatures__[i]
                
        return


    def fit_refinment (self):
        
        if self.__bestformula__ == None:
            raise Exception("Error: model not fitted")

        if self.__getype__ == "gen1":
            self.__refine_gen1__()
        else:
            raise Exception("Error: not implemented")

        return


    def predict (self, x, verbose=0):
        pred_y = []

        if type(x) != np.ndarray:
            raise Exception("Error: x should be a numpy array")

        data = {}
        i = 0
        for c in self.__variables__:
            for bf in self.__variables__[c]:
                if i > x.shape[1]:
                    raise Exception("Error: too many basic features") 

                data[bf] = list(x[:,i])
                i += 1
        
        newf = self.__get_new_feature__(data, self.__bestformula__) 
        if newf[0] == None:
            raise Exception("Error: new feature could not be generated")
        
        pred_y = self.__bestlr__.predict(newf.reshape(-1, 1))

        return np.asarray(pred_y)
    

    def get_formula (self):

        formula = ""
        if self.__getype__ == "gen1":
            if self.__bestformula__ == None:
                raise Exception("Error: model not fitted")

            formula = str("%10.5e"%(self.__bestlr__.intercept_[0]))  + " + (" + \
                str("%10.5e"%(self.__bestlr__.coef_[0][0])) + " * " + \
                    self.__bestformula__ + ")"
            

        return formula

######################################################################

def build_model (gentype, variables):
    
    model = formula_gen (gentype, variables)

    return model 