# MXNet facial-keypoints detection

### 1. About dataset
Dataset [link](https://www.kaggle.com/c/facial-keypoints-detection/data)
### 2. About model
CNN at [model_creating.py](./workspace/model_creating.py) file.

### 3. How to run this tutorial (without TLC and locally as a simulation):

1. Run director:
```sh
cd director_folder
./start_director.sh
```

2. Run envoy:
```sh
cd envoy_folder
./start_envoy.sh env_one shard_config_one.yaml
```

Optional: start second envoy:
 - Copy `envoy_folder` to another place and run from there:
```sh
./start_envoy.sh env_two shard_config_two.yaml
```

3. Run `MXNet_landmarks.ipynb` jupyter notebook:
```sh
cd workspace
jupyter notebook MXNet_landmarks.ipynb
```
