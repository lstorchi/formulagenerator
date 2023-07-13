> runall.out

export count=1
for bf1 in DEmFMO2 F2LE PIEmFMO2 FE2
do
  for bf2  in  mTDS numbtors
  do 
    for bf3 in HIEmligand HIEligandE logP
    do 
      echo "FR3 Counter: " $count $bf1 $bf2 $bf3 >> runall.out  

      python3 ./generatefeats.py  -f cleandataset.xlsx \
        -b "$bf1[1];$bf2[1];$bf3[1]" >> runall.out  
      python3 ffilter.py -t $count -u -f newadata.pkl -n 50 \
        -i "./cleandataset.xlsx,Gexp,nonxtb" >> runall.out  
      
      for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
      do 
         mv $name $name"."$count
      done

      count=$((count+1))

    done
  done
done

export bf4="logP"
export count=1
for bf1 in DEmFMO2 F2LE PIEmFMO2 FE2
do
  for bf2  in  mTDS numbtors
  do 
    for bf3 in HIEmligand HIEligandE
    do 
      echo "FR4 Counter: " $count $bf1 $bf2 $bf3 $bf4 >> runall.out  

      python3 ./generatefeats.py --fourelementsformula -f cleandataset.xlsx \
        -b "$bf1[1];$bf2[1];$bf3[1];$bf4[1]"  >> runall.out   
      python3 ffilter.py -t $count -u -f newadata.pkl -n 50 \
        -i "./cleandataset.xlsx,Gexp,nonxtb" >> runall.out  
      
      for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
      do 
         mv $name $name"."$count
      done

      count=$((count+1))

    done
  done
done

export bf4="dEpligand"
for bf1 in PIEmFMO2 FE2
do
  for bf2  in  mTDS numbtors
  do 
    for bf3 in HIEmligand HIEligandE logP
    do 
      echo "FR4 Counter: " $count $bf1 $bf2 $bf3 $bf4 >> runall.out  

      python3 ./generatefeats.py --fourelementsformula -f cleandataset.xlsx \
        -b "$bf1[1];$bf2[1];$bf3[1];$bf4[1]"  >> runall.out   
      python3 ffilter.py -t $count -u -f newadata.pkl -n 50 \
        -i "./cleandataset.xlsx,Gexp,nonxtb" >> runall.out   
      
      for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
      do 
        mv $name $name"."$count
      done

      count=$((count+1))
    done
  done
done

grep "Max R2 value" runall.out  > allformula.csv 

