# Code for my personal graduation paper

- Codes in `TD3` are adapted from [https://github.com/sfujim/TD3.git](https://github.com/sfujim/TD3.git)
- Codes in `SAC` are adapted from [https://github.com/pranz24/pytorch-soft-actor-critic.git](https://github.com/pranz24/pytorch-soft-actor-critic.git)

## Usage

Start training:
```bash
policy="adv"
# options: td3/adv
cd TD3/ && nohup sh run_experiments.sh ${policy} &
cd ../SAC/ && nohup sh run_experiments.sh ${policy} &
```
This will generate folders including `logs`, `tensorboard`, and `train`,
containing training/testing logs, tensorboard log files, saved models after training, respectively.

Clean all temporary files
```bash
sh clean.sh
```

Run generalization tests:
```bash
nohup sh adv_test_all.sh &
```
This will generate log files in directory `./logs/eval_${name}_all.log`,
where `name` is either `mass` or `noise`, representing NT and HV tests respectively.
