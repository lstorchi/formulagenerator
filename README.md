
export count=1
for bf1 in DEmFMO2 F2LE PIEmFMO2 FE2
do
  for bf2  in  mTDS numbtors
  do 
    for bf3 in HIEmligand HIEligandE logP
    do 
      echo "Counter: " $count
      echo $bf1 $bf2 $bf3

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


export count=1
for bf1 in PIEmFMO2 FE2
do
  for bf2  in  mTDS numbtors
  do 
    for bf3 in HIEmligand HIEligandE logP
    do 
      for bf4 in HIEmligand HIEligandE logP
      do 
        echo "Counter: " $count
        echo $bf1 $bf2 $bf3

        python3 ./generatefeats.py --fourelementsformula -f cleandataset.xlsx -b "$bf1[1];$bf2[1];$bf3[1];$bf4[1]" 
        python3 ffilter.py -u -f newadata.pkl -n 50 -i "./cleandataset.xlsx,Gexp,nonxtb"
      
        for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
        do 
          mv $name $name"."$count
        done

        count=$((count+1))

      done
    done
  done
done

python3 formulaptimizer.py --formula "<3 elements formula>" -f newadata.pkl.xx -i "./cleandataset.xlsx,Gexp,nonxtb"
