$ python3 ./generatefeats.py  -f cleandataset.xlsx -b "FE3[1];numbtors[1];HIEligandE[1];logP[1]" 
$ python3 ffilter.py -f newadata.pkl -n 50 -i "./cleandataset.xlsx,Gexp,nonxtb"
