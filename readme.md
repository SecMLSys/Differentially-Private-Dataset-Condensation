## Differentially Private Dataset Condensation


The DPDC's code is written based on the public dataset condensation codebase from https://github.com/VICO-UoE/DatasetCondensation

Please let us know if you encounter any problem when you run the DPDC's code in the comments.

## Prepare for running the code

```
conda create --name dpdc python=3.7.7

conda activate dpdc

pip install -r requirements.txt
```


## NDPDC

```
python main_dpdc.py  --dataset MNIST  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_scale_rotate  --init noise  --lr_img 1  --num_exp 5  --num_eval 1 --Iteration 10000
```

```
python main_dpdc.py  --dataset FashionMNIST  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init noise  --lr_img 1  --num_exp 5  --num_eval 1 --Iteration 10000
```

```
python main_dpdc.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init noise  --lr_img 1  --num_exp 5  --num_eval 1 --Iteration 10000
```

```
python main_dpdc.py  --dataset CelebA  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init noise  --lr_img 1  --num_exp 5  --num_eval 1 --Iteration 10000
```



## LDPDC

```
python linear_dpdc.py --dataset MNIST --dsa_strategy color_crop_cutout_scale_rotate --ipc 50
```

```
python linear_dpdc.py --dataset FashionMNIST --dsa_strategy color_crop_cutout_flip_scale_rotate --ipc 50
```

```
python linear_dpdc.py --dataset CIFAR10 --dsa_strategy color_crop_cutout_flip_scale_rotate --ipc 50
```

```
python linear_dpdc.py --dataset CelebA --dsa_strategy color_crop_cutout_flip_scale_rotate --ipc 50
```


## Original Distribution Matching Method

```
python main_DM.py  --dataset MNIST  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_scale_rotate  --init noise  --lr_img 1  --num_exp 5  --num_eval 1 

```

```
python main_DM.py  --dataset FashionMNIST  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init noise  --lr_img 1  --num_exp 5  --num_eval 1
```

```
python main_DM.py  --dataset CIFAR10  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init noise  --lr_img 1  --num_exp 5  --num_eval 1
```

```
python main_DM.py  --dataset CelebA  --model ConvNet  --ipc 50  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init noise  --lr_img 1  --num_exp 5  --num_eval 1
```


## Privacy Budget
```
python privacy_budget.py
```