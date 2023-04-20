
export count=1
for bf1 in DEmFMO2 F2LE PIEmFMO2 FE2 PIEmFMO3 FE3
do
  for bf2  in  mTDS numbtors
  do 
    for bf3 in HIEmligand HIEligandE logP
    do 
      echo "Counter: " $count
      python3 ./generatefeats.py  -f cleandataset.xlsx -b "$bf1[1];$bf2[1];$bf3[1]" 
      python3 ffilter.py -u -f newadata.pkl -n 50 -i "./cleandataset.xlsx,Gexp,nonxtb"
      
      for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
      do 
         mv $name $name"."$count
      done

      count=$((count+1))

    done
  done
done


