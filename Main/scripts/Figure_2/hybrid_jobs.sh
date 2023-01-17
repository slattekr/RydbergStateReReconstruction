# #!/bin/bash

# FSS Dataset

joblist=$(sq -h --format="%j")

# for delta in $(seq -1.545 0.5 13.455) # all deltas
for delta in -1.545 4.455 4.955 13.455
do
    for data_epochs in seq(100 100 1000)
    do
        for seed in $(seq 100 100 100) # one seed for now
        do
            X="$delta|OneD|Nh=32|Hybrid_$data_epochs|$seed"
            sbatch -J "$X" --export="delta=$delta,dim=OneD,nh=32,seed=$seed" submit_hybrid_training.sh

            X="$delta|TwoD|Nh=16|Hybrid_$data_epochs|$seed"
            sbatch -J "$X" --export="delta=$delta,dim=TwoD,nh=16,seed=$seed" submit_hybrid_training.sh
                
        done 
        sleep 0.5s
    done
done

