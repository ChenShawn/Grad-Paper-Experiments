#!/bin/bash
set -x

#rm nohup.out
#rm -rf train/*
#rm logs/*

policy="Gaussian"
feval="./logs/evaluate.log"
if [ $# -ge 1 ]
then
    echo "policy: $1"
    policy=$1
fi

for i in `seq 0 9`
do
	echo "Round $i -----------------------------------------"

	python main.py \
	--policy ${policy} \
	--env "HalfCheetah-v2" \
	--seed $i \
	--num-steps 3000000 \
	--start-steps 10000 \
    --use-adv &

	python main.py \
	--policy ${policy} \
	--env "Hopper-v2" \
	--seed $i \
	--num-steps 3000000 \
	--start-steps 10000 \
    --use-adv &

	python main.py \
	--policy ${policy} \
	--env "Walker2d-v3" \
	--seed $i \
	--num-steps 3000000 \
	--start-steps 10000 \
    --use-adv &

	python main.py \
	--policy ${policy} \
	--env "Ant-v2" \
	--seed $i \
	--num-steps 3000000 \
	--start-steps 100000 \
    --use-adv &

	python main.py \
	--policy ${policy} \
	--env "Humanoid-v2" \
	--seed $i \
	--num-steps 3000000 \
	--batch-size 512 \
	--start-steps 10000 \
    --use-adv &

	python main.py \
	--policy ${policy} \
	--env "LunarLanderContinuous-v2" \
	--seed $i \
	--start-steps 10000 \
    --use-adv &

	wait

	python test.py -p ${policy} -e "Hopper-v2" -n 100 --train-seed $i --test-seed $i >> ${feval} &
	python test.py -p ${policy} -e "HalfCheetah-v2" -n 100 --train-seed $i --test-seed $i >> ${feval} &
	python test.py -p ${policy} -e "Walker2d-v3" -n 100 --train-seed $i --test-seed $i >> ${feval} &
	python test.py -p ${policy} -e "Ant-v2" -n 100 --train-seed $i --test-seed $i >> ${feval} &
	python test.py -p ${policy} -e "LunarLanderContinuous-v2" -n 100 --train-seed $i --test-seed $i >> ${feval} &
	python test.py -p ${policy} -e "Humanoid-v2" -n 100 --train-seed $i --test-seed $i >> ${feval} &

    wait
done

sh adv_test_all.sh
