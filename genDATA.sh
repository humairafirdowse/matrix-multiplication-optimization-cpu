#!/bin/bash
#Generate data.txt
#May need to set execute permission: "chmod +x genDATA.sh"
matrix_sizes=(32 64 128 255 256 510 511 512 513 768 769 1023 1024 1025 1033 2047 2048 2049)


> data.txt

for size in "${matrix_sizes[@]}"
do
	printf 'Testing N=%d\t' $size
	SUM=0
	MAX_GFLOPS=0
	for i in {1..20}
	do
		OUTPUT=$(./benchmark-blas -n $size -g)
		GFLOPS=$(echo $OUTPUT | awk '{print $2}')
		SUM=$(awk '{print $1+$2}' <<<"${SUM} ${GFLOPS}")
        	if (( $(echo "$GFLOPS > $MAX_GFLOPS" | bc -l) )); then
            		MAX_GFLOPS=$GFLOPS
        	fi
	done
	#AVG=$(awk '{print $1/20}' <<<"${SUM}")
	# printf 'AVG %s GFLOPS\n'  $AVG 
	# printf '%d\t%s\n' $size $AVG >> data.txt
	printf 'MAX %s GFLOPS\n' $MAX_GFLOPS
    printf '%d\t%s\n' $size $MAX_GFLOPS >> data.txt
done

printf 'Results saved in data.txt\n'
