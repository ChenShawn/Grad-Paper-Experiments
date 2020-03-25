set -x

rm -rf tensorboard
rm -rf logs
rm -rf train

cache_files=`find -name __pycache__`
for cf in ${cache_files}
do
    rm -rf ${cf}
done
rm nohup.out
