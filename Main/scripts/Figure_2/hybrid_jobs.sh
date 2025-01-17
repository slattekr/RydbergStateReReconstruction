# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

# for delta in $(seq -1.545 0.5 13.455) # all deltas
for delta in 13.455  #-1.545 4.455 4.955 #13.455
do
    for data_epochs in $(seq 200 200 1200)
    do
        for seed in $(seq 100 100 100) # one seed for now
        do
            X="d=$delta|OneD|Nh=32|Hybrid_$data_epochs|$seed"
            sbatch -J "$X" --export="delta=$delta,data_epochs=$data_epochs,dim=OneD,nh=32,seed=$seed" submit_hybrid_training.sh

            #X="$delta|TwoD|Nh=16|Hybrid_$data_epochs|$seed"
            #sbatch -J "$X" --export="delta=$delta,data_epochs=$data_epochs,dim=TwoD,nh=16,seed=$seed" submit_hybrid_training.sh
                
        done 
        sleep 0.5s
    done
done

