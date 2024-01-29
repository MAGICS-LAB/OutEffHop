# Outlier Efficient Modern Hopfield Model (OutEffHop)

This is the code of the paper OutHopEff. You can use this repo to reproduce the results in the paper.

## Outline Efficiency of OutEffHop
### Environmental Setup
You can set up the experimental environment by running the following command line:


```shell
Set locale variables and add the project root directory to your pythonpath:

export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export PYTHONPATH=${PYTHONPATH}:$(realpath "$PWD")
```

Create suitable environment for different experiment.  

For experiment in paper section 4.1, outlier efficiency of BERT and OPT:
```bash
conda create -n outlier python==3.9

# Run the pip module as a script.
python -m pip install -r /your_path/OutEffHop/OutEffHop/requirements.txt
```
For experiment in paper section 4.1, quantization of StanHop :

```bash
conda create -n quantize_STHM python==3.8

# Run the pip module as a script.
python -m pip install -r /your_path/OutEffHop/OutEffHop/STanHop_outlier/quantize_requirements.txt
```

For the other experiment in paper 4.1, 4.2 about STanHop : 
```bash
conda create -n STHM python==3.8

# Run the pip module as a script.
python -m pip install -r /your_path/OutEffHop/STanHop_time_seeries/requirements.txt
```
### Script for experiments in paper 4.1, 4.2

We upload all the experiments slurm script in `\OutEffHop`.






## Experimental Validation of Theoretical Results
### Environmental Setup

You can set up the experimental environment by running the following command line:

```shell
$ conda create -n theory_verify python=3.8
$ conda activate theory_verify
$ pip3 install -r requirements.txt
```



### Plotting

```shell
$ cd theory_verfication/
$ python3 Plotting.py
```


## OutEffHop Case study

### Environmental Setup

You can set up the experimental environment by running the following command line:


```shell
$ cd STanHop_time_seeries
$ pip3 install -r requirements.txt
$ export PYTHONPATH=$PYTHONPATH:$PWD
```



### Reproducibility
1. Put datasets to conduct experiments into folder `datasets/`. We have already put `ETTh1` and `ETTm1` into it. `WTH` and `ECL` can be downloaded from 
https://github.com/zhouhaoyi/Informer2020. `ILI` and `Traffic` can be downloaded from https://github.com/thuml/Autoformer. Note that the `WTH` we used in the paper is the one with 12 dimensions from Informer, not the one with 21 dimensions from Autoformer.

2. To get results of Crossformer with $T=168, \tau = 48, L_{seg} = 6$ on ETTh1 dataset, run:
```
python main_stanhop.py  --data ETTh1 --in_len 168 --out_len 48 --seg_len 6 --learning_rate 1e-4 --itr 1 --mode softmax1 --use_gpu --gpu 0  --batch_size 128 --run_name STHM_softmax1  --e_layers 11
```
The model will be automatically trained and tested. The trained model will be saved in folder `checkpoints/` and evaluated metrics will be saved in folder `results/`.

4. To reproduce all results in the paper, run following scripts to get corresponding results:
```
batch OutEffHop_script/submit_STHM.sh
```


`main_stanhop` is the entry point of our model and there are other parameters that can be tuned. Here we describe them in detail:
| Parameter name | Description of parameter |
| --- | --- |
| data           | The dataset name                                             |
| root_path      | The root path of the data file (defaults to `./datasets/`)    |
| data_path      | The data file name (defaults to `ETTh1.csv`)                  |
| data_split | Train/Val/Test split, can be ratio (e.g. `0.7,0.1,0.2`) or number (e.g. `16800,2880,2880`), (defaults to `0.7,0.1,0.2`) 
| checkpoints    | Location to store the trained model (defaults to `./checkpoints/`)  |
| in_len | Length of input/history sequence, i.e. $T$ in the paper (defaults to 96) |
| out_len | Length of output/future sequence, i.e. $\tau$ in the paper (defaults to 24) |
| seg_len | Length of each segment in DSW embedding, i.e. $L_{seg}$ in the paper (defaults to 6) |
| win_size | How many adjacent segments to be merged into one in segment merging of HED  (defaults to 2) |
| factor | Number of routers in Cross-Dimension Stage of TSA, i.e. $c$ in the paper (defaults to 10) |
| data_dim | Number of dimensions of the MTS data, i.e. $D$ in the paper (defaults to 7 for ETTh and ETTm) |
| d_model | Dimension of hidden states, i.e. $d_{model}$ in the paper (defaults to 256) |
| d_ff | Dimension of MLP in MSA (defaults to 512) |
| n_heads | Num of heads in MSA (defaults to 4) |
| e_layers | Num of encoder layers, i.e. $N$ in the paper (defaults to 3) |
| dropout | The probability of dropout (defaults to 0.2) |
| weight_decay| The weight decay
| num_workers | The num_works of Data loader (defaults to 0) |
| batch_size | The batch size for training and testing (defaults to 32) |
| train_epochs | Train epochs (defaults to 20) |
| patience | Early stopping patience (defaults to 3) |
| learning_rate | The initial learning rate for the optimizer (defaults to 1e-4) |
| lradj | Ways to adjust the learning rate (defaults to `type1`) |
| itr | Experiments times (defaults to 1) |
| save_pred | Whether to save the predicted results. If True, the predicted results will be saved in folder `results` in numpy array form. This will cost a lot time and memory for datasets with large $D$. (defaults to `False`). |
| use_gpu | Whether to use gpu (defaults to `True`) |
| gpu | The gpu no, used for training and inference (defaults to 0) |
| use_multi_gpu | Whether to use multiple gpus (defaults to `False`) |
| devices | Device ids of multile gpus (defaults to `0,1,2,3`) |
| mode | The type of the Hopfield Network  (Hopfield, SparseHopfield, STanHop, OutEffHop) |
| run_name | The name of experiment |
| eta | The eta value of Entmax |
| gamma | The gamma value of Entmax |

## Acknowledgment

The experiments in this work benefit from the following open-source codes:

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/MAGICS-LAB/STanHop

https://github.com/Qualcomm-AI-research/outlier-free-transformers



