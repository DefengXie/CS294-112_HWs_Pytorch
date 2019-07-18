set -eux
mkdir -p experiments
rm -rf ./experiments/*
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2
do
    python3 main.py --env_name $e --algorithm behavioral_cloning
    python3 main.py --env_name $e --algorithm dagger --num_epochs 300
done