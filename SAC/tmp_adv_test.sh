set -x

seed=0
env_name="HalfCheetah-v2"
if [ $# -ge 2 ]
then
    seed=$1
    env_name=$2
    echo "radnom seed: ${seed}"
else
    echo "Usage: sh tmp_adv_test.sh \${seed} \${env_name}"
    exit -1
fi

mass_log="./logs/eval_mass_all.log"
noise_log="./logs/eval_noise_all.log"

# use the random seed with the best performance on vanilla td3 to test the performance of adv models
# try test using more train seeds later

# relative mass test
for mass in `seq 0.5 0.05 1.5001`
do
    python test.py -e ${env_name} -n 100 --train-seed 123456 --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
    #python test.py -p adv -e Hopper-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
    #python test.py -p adv -e Walker2d-v3 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
    #python test.py -p adv -e Ant-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
    #python test.py -p adv -e Humanoid-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
    #python test.py -p adv -e BipedalWalker-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --relative-mass ${mass} >> ${mass_log}
done


# noise scale test
for scale in `seq 0.0 0.05 0.5`
do
    python test.py -e ${env_name} -n 100 --train-seed 123456 --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
    #python test.py -p adv -e Hopper-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
    #python test.py -p adv -e Walker2d-v3 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
    #python test.py -p adv -e Ant-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
    #python test.py -p adv -e Humanoid-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
    #python test.py -p adv -e BipedalWalker-v2 -n 100 --train-seed ${seed} --test-seed ${seed} --noise-scale ${scale} >> ${noise_log}
done

