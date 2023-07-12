import pandas as pd
import numpy as np

#import matplotlib
#import matplotlib.pyplot as plt
#import seaborn as sns

import argparse
import math
import sys
import os

import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("./common/")

import generators

# get from here https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop

def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]

if __name__ == "__main__":
    tabtoread = "nonxtb"
    quiet = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file", help="input xlsx file ", \
                        required=False, default="", type=str)
    parser.add_argument("-c","--csvfile", help="input csv file ", \
                        required=False, default="", type=str)
    parser.add_argument("-b","--basicfeatures", \
                        help="input ; separated list of basic featuresto combine \n" + \
                        "   each feature has an associated type (i.e. \"IP[1];EA[1];Z[2]\"", \
                        required=True, type=str)
    parser.add_argument("-d", "--dumponly", \
                        help="to dump only the first N formulas",
                        required=False, type=int, default=-1)
    parser.add_argument("-v", "--verbose", \
                        help="verbose mode", action="store_true",
                        required=False, default=False)
    parser.add_argument("-r", "--reducemem", \
                        help="use less memory for correlation", action="store_true",
                        required=False, default=False)
    parser.add_argument("--fourelementsformula", \
                        help="Generate formulas including four elements", action="store_true",
                        required=False, default=False)
    parser.add_argument("-j", "--jumpremoving", \
                        help="Do not filter the features considering the correlation", action="store_true",
                        required=False, default=False)
    parser.add_argument("-t", "--tabtouse", \
                        help="Name of the tab to be used", type=str, required=False, \
                        default=tabtoread)
    
    args = parser.parse_args()
    
    xslxfilename = args.file
    csvfilename = args.csvfile
    tabtoread = args.tabtouse

    data = None
    if xslxfilename != "":
        xls = pd.ExcelFile(xslxfilename)
        data = pd.read_excel(xls, tabtoread, index_col=0)
    elif csvfilename != "":
        data = pd.read_csv(csvfilename, index_col=0)

    if data is None:
        print("Input filename not specified")
        exit(1)
    
    basicfeatureslist = args.basicfeatures.split(";")

    #for V_alpha in data["V_alpha"]:
    #   print(V_alpha)
    #   print(math.exp(V_alpha))

    corrlimit = 0.95

    basicfeaturesdict = {}
    featureslist = []
    for b in basicfeatureslist:
        newb = b.split("[")
        if len(newb) != 2:
            print("Error in basicfeatures format")
            exit(1)
            
        classe = newb[1].replace("]", "")
        name = newb[0]
        
        if not (name in data.columns):
            print("Error feature ", name, "not present")
            for bf in data.columns:
                print("   ", bf )
            exit(1)

        featureslist.append(name)
            
        if not (classe in basicfeaturesdict):
            basicfeaturesdict[classe] = []
            basicfeaturesdict[classe].append(name)
        else:
            basicfeaturesdict[classe].append(name)

    atleastone = False
    if not quiet: 
        print("Top Absolute Correlations")
    topcorr = get_top_abs_correlations(data, 10)
    for key, value in topcorr.items():
       #print("%30s %30s %10.6f"%(key[0], key[1], value))
       if (value > corrlimit):
          if (key[0] in featureslist) or (key[1] in featureslist):
            if not quiet: 
                print("High correlationd found ", key[0] ," and ", key[1])

    #corrmat = data.corr()
    #ax = sns.heatmap(corrmat, annot = True, vmin=-1, vmax=1, center= 0)
    #ax = sns.heatmap(corrmat, vmin=-1, vmax=1, center= 0)
    #plt.savefig("cooheatmap.png")
    #plt.show()

    basicfeaturesdict = {}
    for b in basicfeatureslist:
        newb = b.split("[")
        if len(newb) != 2:
            print("Error in basicfeatures format")
            exit(1)
            
        classe = newb[1].replace("]", "")
        name = newb[0]
        
        if not (name in data.columns):
            print("Error feature ", name, "not present")
            for bf in data.columns:
                print("   ", bf )
            exit(1)
            
        if not (classe in basicfeaturesdict):
            basicfeaturesdict[classe] = []
            basicfeaturesdict[classe].append(name)
        else:
            basicfeaturesdict[classe].append(name)
    
    try:
        if not quiet: 
            print("Start generating formulas...")
        formulas = None
        if args.fourelementsformula:
            formulas = generators.generate_formulas_four (basicfeaturesdict)
        else:
            formulas = generators.generate_formulas (basicfeaturesdict)
        fname  = "formulaslist.txt"
        if os.path.exists(fname):
            os.remove(fname)
        fp = open(fname, "w")
        for f in formulas:
            fp.write(f + "\n")
        fp.close()
        if not quiet: 
            print("Generated ", len(formulas) ," formulas...")
        if not quiet: 
            print ("Start generating features...")
        last = args.dumponly

        max = last  
        if last < 0:
            max = len(formulas)

        newdataframe = {}
        
        for idx, formula in enumerate(formulas[0:last]):

            if args.verbose:
                print ("%10d of %10d"%(idx+1, max))
                sys.stdout.flush()
            else:
                if not quiet:
                    generators.progress_bar(idx+1, max)
            
            newf = None

            try:
                newf = generators.get_new_feature(data, formula)
            except OverflowError:
                print("Math error in formula (overflow)", formula)
            except ZeroDivisionError:
                print("Math error in formula (division by zero)", formula)

            if newf is not None:
                newdataframe[formula] = newf
        
        if not args.verbose:
            if not quiet: 
                print()
            
        newatomicdata = pd.DataFrame.from_dict(newdataframe)   
        if not quiet:   
            print ("Produced ", newatomicdata.size , " data features")
        corrlimit = 0.98

        if not args.jumpremoving:
            if not quiet: 
                print ("Start removing highly correlated features (limit: %10.5f)"%corrlimit)
            if args.reducemem:
                cname = list(newatomicdata.columns)
                
                to_drop = []
                for i in range(len(cname)):
                    if not args.verbose:
                        if not quiet: 
                            generators.progress_bar(i+1, len(cname))
                    else:
                        if not quiet: 
                            print("%10d of %1dd"%(i+1, len(cname)))
                            sys.stdout.flush()
                    f1 = cname[i]
                    for j in range(i+1,len(cname)):
                        f2 = cname[j]
                        if not (f2 in to_drop):
                            corrvalue = abs(newatomicdata[f1].corr(newatomicdata[f2], \
                                                           method='pearson'))
                        
                            #print(f1, f2, corrvalue)
                            if (corrvalue > corrlimit):
                                to_drop.append(f2)
                                break
            
                if not args.verbose:
                    if not quiet: 
                        print("")
                if not quiet: 
                    print("  Removing ", len(to_drop), " features")
                
                if args.verbose:
                    for f in to_drop:
                        if not quiet: 
                            print("    ", f)
                
                newatomicdata = newatomicdata.drop(newatomicdata[to_drop], axis=1)
                if not quiet: 
                    print ("Produced ", newatomicdata.size , " data features")
            else:
                corr = newatomicdata.corr(method = 'pearson').abs()
                            
                # Select upper triangle of correlation matrix
                upper = corr.where(np.triu(\
                    np.ones(corr.shape), k=1).astype(bool))
                to_drop = [column for column in \
                    upper.columns if any(upper[column] > corrlimit)]
                if not quiet: 
                    print("  Removing ", len(to_drop), " features")
            
                if args.verbose:
                    for f in to_drop:
                        if not quiet: 
                            print("    ", f)
            
                newatomicdata = newatomicdata.drop(newatomicdata[to_drop], axis=1)
                if not quiet: 
                    print ("Produced ", newatomicdata.size , " data features")
        
                #if (args.verbose):
                #    corr = newatomicdata.corr(method = 'pearson').abs()
                #    scorr = corr.unstack()
                #    so = scorr.sort_values(kind="quicksort")
                
                #    for index, value in so.items():
                #        if index[0] != index[1]:
                #            print (index[0], index[1], " ==> ", value)
                
                fname = "finalformulalist.txt"
                if os.path.exists(fname):
                    os.remove(fname)
                fp = open(fname, "w")
                for f in newatomicdata:
                    fp.write(f + "\n")
                fp.close()
        
        
        newatomicdata.to_pickle("newadata.pkl")
        newatomicdata.to_csv("newadata.csv")
        
        #plt.figure(figsize=(12,10))
        #cor = newatomicdata.corr()
        #sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        #plt.show()
        
    except NameError as err:
        print(err)


