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

        self.__bestformula__ = None
        self.__bestlr__ = None
        self.__bestnewf__ = None
        self.__bestmse__ = None


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
    
    def __refine_gen1__ (self, x, y, numofinterval, roundto):

        if self.__bestformula__ == None:
            raise Exception("Error: model not fitted")
        
        #print("best formula: ", self.__bestformula__)
        #print("best rmse: ", self.__bestmse__)

        num = self.__bestformula__.split("/")[0]
        denum = self.__bestformula__.split("/")[1]

        a = ""
        b = ""
        op = ""
        if num.find(" + ") != -1:
            a = num.split(" + ")[0]
            b = num.split(" + ")[1]
            op = " + "
        elif num.find(" - ") != -1:
            a = num.split(" - ")[0]
            b = num.split(" - ")[1]
            op = " - "
        elif num.find(" * ") != -1:
            a = num.split(" * ")[0]
            b = num.split(" * ")[1]
            op = " * "
        
        a = a.replace(" ", "")
        b = b.replace(" ", "")
        denum = denum.replace(" ", "")

        a = a[1:]
        b = b[:-1]
        denum = denum[1:-1]

        #print("a: ", a)
        #print("b: ", b)
        #print("denum: ", denum)

        foundabestone = False
        dx = 1.0/float(numofinterval)
        aw = 0.0
        for ia in range(numofinterval):
            aw = aw + dx
            bw = 0.0
            for ib in range(numofinterval):
                bw = bw + dx
                denumw = 0.0
                for idenum in range(numofinterval):
                    denumw = denumw + dx

                    if aw == bw and aw == denumw:
                        continue

                    #round numbers to 2 decimal places
                    aw = round(aw, roundto)
                    bw = round(bw, roundto)
                    denumw = round(denumw, roundto)

                    newformula = "(" + str(aw) + " * " + a + ") " + op + " (" + \
                        str(bw) + " * " + b + ") / (" + \
                        str(denumw) + " * " + denum + ")"
                    #print("new formula: ", newformula)

                    data = {}
                    i = 0
                    for c in self.__variables__:
                        for bf in self.__variables__[c]:
                            if i > x.shape[1]:
                                raise Exception("Error: too many basic features") 

                            data[bf] = list(x[:,i])
                            i += 1
                    
                    newf = self.__get_new_feature__(data, newformula)

                    lr = LinearRegression()
                    lr.fit(newf.reshape(-1, 1), y)
                    mse = np.mean((lr.predict(newf.reshape(-1, 1)) - y) ** 2)
                    #print("new formula: ", newformula, " mse: ", mse)
                    if mse < self.__bestmse__:
                        self.__bestmse__ = mse
                        self.__bestformula__ = newformula
                        self.__bestlr__ = lr
                        self.__bestnewf__ = newf
                        foundabestone = True
        
        #print("best formula: ", self.__bestformula__)
        #print("best rmse: ", self.__bestmse__)
        #raise Exception("Error: not implemented")

        return foundabestone
    

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
        newfeatures = []
        for idxf, f in enumerate(self.__formulas__):
            newf = self.__get_new_feature__(data, f) 
            if newf[0] == None:
                fidxtorm.append(idxf)
                #print(" torm :", f)
                newfeatures.append([])
            else:
                newfeatures.append(newf)
                #print(f)
                #for i in range(len(newf)):
                #    v = data["v"][i]
                #    cE = data["cE"][i]
                #    print(v, cE)
                #    print((v+cE) / math.exp(v),  newf[i])

        for index in sorted(fidxtorm, reverse=True):
            del newfeatures[index]
            del self.__formulas__[index]

        self.__bestmse__ = float("inf") 
        for i in range(len(newfeatures)):
            lr = LinearRegression()
            newf = newfeatures[i]
            lr.fit(newf.reshape(-1, 1), y)
            mse = np.mean((lr.predict(newf.reshape(-1, 1)) - y) ** 2)
            if mse < self.__bestmse__:
                self.__bestmse__ = mse
                self.__bestformula__ = self.__formulas__[i]
                self.__bestlr__ = lr
                self.__bestnewf__ = newf

        return


    def fit_refinment (self, x, y, numofinterval=10, roundto=2):
        
        if self.__bestformula__ == None:
            raise Exception("Error: model not fitted")

        if self.__getype__ == "gen1":
            return self.__refine_gen1__(x, y, numofinterval, roundto)
        else:
            raise Exception("Error: not implemented")

        return False


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