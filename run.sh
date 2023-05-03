# 3 one for each

#export count=1
#for bf1 in DEmFMO2 F2LE PIEmFMO2 FE2 PIEmFMO3 FE3
#do
#  for bf2  in mTDS numbtors
#  do 
#    for bf3 in HIEmligand HIEligandE logP
#    do 
#      echo "Counter: " $count
#      echo $bf1 $bf2 $bf3
#
#      python3 ./generatefeats.py  -f cleandataset.xlsx -b "$bf1[1];$bf2[1];$bf3[1]" 
#      python3 ffilter.py -u -f newadata.pkl -i "./cleandataset.xlsx,Gexp,nonxtb"
#      
#      for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
#      do 
#         mv $name $name"."$count
#      done
#
#      count=$((count+1))
#
#    done
#  done
#done


# 3 elements alla possible

#export count=1
#for bf1 in DEmFMO2 F2LE PIEmFMO2 FE2 PIEmFMO3 FE3 mTDS numbtors HIEmligand HIEligandE logP
#do
#  for bf2  in DEmFMO2 F2LE PIEmFMO2 FE2 PIEmFMO3 FE3 mTDS numbtors HIEmligand HIEligandE logP
#  do 
#    if [ "$bf1" != "$bf2" ]; then
#      for bf3 in DEmFMO2 F2LE PIEmFMO2 FE2 PIEmFMO3 FE3 mTDS numbtors HIEmligand HIEligandE logP
#      do 
#        if [ "$bf3" != "$bf1" ]; then
#          if [ "$bf3" != "$bf2" ]; then
#            echo "Counter: " $count
#            echo $bf1 $bf2 $bf3
#           
#            python3 ./generatefeats.py  -f cleandataset.xlsx -b "$bf1[1];$bf2[1];$bf3[1]" 
#            python3 ffilter.py -u -f newadata.pkl -i "./cleandataset.xlsx,Gexp,nonxtb"
#            
#            for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
#            do 
#               mv $name $name"."$count
#            done
#           
#            count=$((count+1))
#          fi
#        fi
#      done
#    fi
#  done
#done

#export count=1
#for bf1 in DG LE
#do
#  for bf2 in HIEmligand
#  do 
#    for bf3 in logP
#    do 
#      echo "Counter: " $count
#      echo $bf1 $bf2 $bf3
#
#      python3 ./generatefeats.py -t "xtb" -f cleandataset.xlsx -b "$bf1[1];$bf2[1];$bf3[1]" 
#      python3 ffilter.py -u -f newadata.pkl -i "./cleandataset.xlsx,Gexp,xtb"
#      
#      for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
#      do 
#         mv $name $name"."$count
#      done
#
#      count=$((count+1))
#
#    done
#  done
#done

export count=1
export bf4="logP"
for bf1 in DEmFMO2 F2LE PIEmFMO2 FE2 PIEmFMO3 FE3
do
  for bf2  in mTDS numbtors
  do 
    for bf3 in HIEmligand HIEligandE 
    do 
      echo "Counter: " $count
      echo $bf1 $bf2 $bf3 $bf4

      python3 ./generatefeats.py -f cleandataset.xlsx --fourelementsformula -b "$bf1[1];$bf2[1];$bf3[1];$bf4[1]" 
      python3 ffilter.py -u -f newadata.pkl -i "./cleandataset.xlsx,Gexp,xtb"

      for name in newadata.pkl finalformulalist.txt newadata.csv feature_mse.csv finalselectedformulas.txt
      do 
         mv $name $name"."$count
      done

      count=$((count+1))

    done
  done
done


