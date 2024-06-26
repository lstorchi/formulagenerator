import os
import sys
import subprocess

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: ", sys.argv[0], " filename.csv ")

    fp = open(sys.argv[1])

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
                "newadata.pkl."+num+" -i " +  "\"./datatousenew.xlsx,Gexp,Foglio1\""
        elif numofelements == 4:
            toexec = "python3 ./formulaptimizer.py --fourelementsformula --formula " + "\""+formula+"\" -f" \
                "newadata.pkl."+num+" -i " +  "\"./datatousenew.xlsx,Gexp,Foglio1\""
        
        os.system(toexec)

