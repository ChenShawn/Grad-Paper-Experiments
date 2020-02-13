#!/bin/bash
set -x

#rm nohup.out
#rm -rf train/*
#rm logs/*

policy="adv"
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
	--max-timesteps 3000000 \
	--start-timesteps 10000 &

	python main.py \
	--policy ${policy} \
	--env "Hopper-v2" \
	--seed $i \
	--max-timesteps 3000000 \
	--start-timesteps 1000 &

	python main.py \
	--policy ${policy} \
	--env "Walker2d-v3" \
	--seed $i \
	--max-timesteps 3000000 \
	--start-timesteps 1000 &

	python main.py \
	--policy ${policy} \
	--env "Ant-v2" \
	--seed $i \
	--max-timesteps 3000000 \
	--start-timesteps 10000 &

	wait
	python test.py -p ${policy} -e "Hopper-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -p ${policy} -e "HalfCheetah-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -p ${policy} -e "Walker2d-v3" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -p ${policy} -e "Ant-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &

	python main.py \
	--policy ${policy} \
	--env "Humanoid-v2" \
	--seed $i \
	--max-timesteps 3000000 \
	--batch-size 512 \
	--start-timesteps 10000 &

	python main.py \
	--policy ${policy} \
	--env "LunarLanderContinuous-v2" \
	--seed $i \
	--start-timesteps 1000 &

	python main.py \
	--policy ${policy} \
	--env "BipedalWalker-v2" \
	--batch-size 1024 \
	--seed $i \
	--start-timesteps 1000 &

	python main.py \
	--policy ${policy} \
	--env "BipedalWalkerHardcore-v2" \
	--batch-size 1024 \
	--seed $i \
	--start-timesteps 1000 &

	wait
	python test.py -p ${policy} -e "LunarLanderContinuous-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -p ${policy} -e "Humanoid-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -p ${policy} -e "BipedalWalker-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
	python test.py -p ${policy} -e "BipedalWalkerHardcore-v2" -n 100 --train-seed $i --test-seed $i >> ./logs/evaluate.log &
done
