# CS294-112 HW 1: Imitation Learning

Dependencies:
  * Python 3.6
  * Pytorch 1.1.0
  * tensorflow-gpu 1.13.1
  * MuJoCo version v2.0 and 2.0.2.4
  * gym 0.13.1


To generate the data for training and testing,run<br>
```
bash get_data.sh
```
To run all experiments , run

```bash
bash run_experiment.sh
```

All results would be saved in `experiments/result.json`


|Task|Algorithm|Mean return |STD |Mean return(expert) |STD(expert)|
|---|---|---|---|---|---|
|Hopper-v2|behavioral_cloning|1931.53|410.99|3777.46|2.81|
|Hopper-v2|dagger|3778.73|2.53|
|Ant-v2|behavioral_cloning|4526.86|156.66|4678.38|418.60|
|Ant-v2|dagger|4810.88|146.48|
|HalfCheetah-v2|behavioral_cloning|4059.39|106.75|4141.06|67.00|
|HalfCheetah-v2|dagger|4192.66|90.06|
|Humanoid-v2|behavioral_cloning|966.03|353.56|10023.57|2259.72|
|Humanoid-v2|dagger|10564.71|41.69|
|Reacher-v2|behavioral_cloning|-7.55|2.32|-3.47|1.61|
|Reacher-v2|dagger|-3.96|1.42|
|Walker2d-v2|behavioral_cloning|1489.10|521.12|5497.49|85.13|
|Walker2d-v2|dagger|5549.87|27.95|
