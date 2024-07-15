import os
import sys
import subprocess

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("usage: ", sys.argv[0], " filename.csv cleandataset.xlsx sheetname")

    fp = open(sys.argv[1])
    execelfnam = sys.argv[2]
    sheetname = sys.argv[3]

    for line in fp:
        sline = line.split(",") 
        num = sline[0]
        formula = sline[-1]
        formula = formula.replace("\n", "")
        formula = formula.lstrip()
        formula = formula.rstrip()

        numofelements = int(sline[1])

        toexe = ""
        if numofelements == 3:
            toexec = "python3 ./formulaptimizer.py --formula " + "\""+formula+"\" -f" \
                "newadata.pkl."+num+" -i " +  "\""+execelfnam+",Gexp,"+sheetname+"\""
        elif numofelements == 4:
            toexec = "python3 ./formulaptimizer.py --fourelementsformula --formula " + "\""+formula+"\" -f" \
                "newadata.pkl."+num+" -i " +  "\""+execelfnam+",Gexp,"+sheetname+"\""
        
        os.system(toexec)

