set -x

mass_log="./logs/eval_mass_all.log"
noise_log="./logs/eval_noise_all.log"

# use the random seed with the best performance on vanilla td3 to test the performance of adv models
# try test using more train seeds later
for seed in `seq 0 9`
do
    
    # relative mass test
    for mass in `seq 0.5 0.05 1.5001`
    do
        python test.py -p td3 -e HalfCheetah-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
        python test.py -p td3 -e Hopper-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
        python test.py -p td3 -e Walker2d-v3 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
        python test.py -p td3 -e Ant-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
        python test.py -p td3 -e Humanoid-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
        python test.py -p td3 -e LunarLanderContinuous-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
    done


    # noise scale test
    for scale in `seq 0.0 0.05 0.5`
    do
        python test.py -p td3 -e HalfCheetah-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
        python test.py -p td3 -e Hopper-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
        python test.py -p td3 -e Walker2d-v3 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
        python test.py -p td3 -e Ant-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
        python test.py -p td3 -e Humanoid-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
        python test.py -p td3 -e LunarLanderContinuous-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
    done

done
