# FusionRetro: Molecule Representation Fusion via In-Context Learning for Retrosynthetic Planning  

This repository contains an implementation of ["FusionRetro: Molecule Representation Fusion via In-Context Learning for Retrosynthetic Planning"](https://openreview.net/pdf?id=cnILy0dQUr), which is an autoregressive framework for molecule synthetic route generation.  



## Dropbox 

We provide the starting material file in dropbox, you can download this file via: 
https://www.dropbox.com/s/nwh2ijrjzbyia73/zinc_stock_17_04_20.hdf5?dl=0  
Please move this file into the root folder.  




## FusionRetro 
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 FusionRetro/  
cd FusionRetro  

#Data Process  
python to_canolize.py  

#Initial Train
python train.py --batch_size 64 --epochs 3000  
# After 3000 epochs, We set global_step to 1000000 and continue to train the model (3000th epoch's model paramater) with 1000 epochs  
#Continue Train
python train.py --batch_size 64 --continue_train --epochs 1000

# We select the model with the performance on the first 100 routes in the validation dataset

#Retro Star Zero Search
python retro_star_0.py  --beam_size 5  

#Retro Star Search
python get_reaction_cost.py  
python get_molecule_cost.py  
python value_mlp.py  
python retro_star.py --beam_size 5  

#Greedy DFS Search
python greedy_dfs.py --beam_size 5  
```



## Transformer  
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 Transformer/  
cd Transformer  

#Data Process  
python to_canolize.py  

#Train  
python train.py --batch_size 32 --epochs 2000  

# We select the model with the performance on the first 100 routes in the validation dataset

#Retrosynthesis Test
python retrosynthesis_test.py --beam_size 10  

#Retro Star Zero Search
python retro_star_0.py  --beam_size 5  

#Retro Star Search
python get_reaction_cost.py  
python get_molecule_cost.py  
python value_mlp.py  
python retro_star.py --beam_size 5  

#Greedy DFS Search
python greedy_dfs.py --beam_size 5  
```


## Retrosim  
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 Retrosim/  
cd Retrosim  

#Retrosynthesis Test
python retrosynthesis_test.py --beam_size 10 --num_cores 64  

#Retro Star Zero Search
python retro_star_0.py  --beam_size 5 --num_cores 64  

#Retro Star Search
python get_reaction_cost.py  
python get_molecule_cost.py  
python value_mlp.py  
python retro_star.py --beam_size 5 --num_cores 64  

#Greedy DFS Search
python greedy_dfs.py --beam_size 5 --num_cores 64  
```



## Neuralsym  
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 Neuralsym/  
cd Neuralsym  

#Data Process
python prepare_data.py  

#Train
bash train.sh  

# We select the model by the original' code's setting 

#Retrosynthesis Test
python retrosynthesis_test.py --beam_size 10 --num_cores 64  

#Retro Star Zero Search
python retro_star_0.py  --beam_size 5 --num_cores 64  

#Retro Star Search
python get_reaction_cost.py  
python get_molecule_cost.py  
python value_mlp.py  
python retro_star.py --beam_size 5 --num_cores 64  

#Greedy DFS Search
python greedy_dfs.py --beam_size 5 --num_cores 64  
```



## GLN  
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 GLN/gln/  
cd GLN  
pip install -e .  
cd gln  

#Data Process 
python process_data_stage_1.py -save_dir data  

python process_data_stage_2.py -save_dir data -num_cores 12 -num_parts 1 -fp_degree 2 -f_atoms data/atom_list.txt -retro_during_train False $@  

python process_data_stage_2.py -save_dir data -num_cores 12 -num_parts 1 -fp_degree 2 -f_atoms data/atom_list.txt -retro_during_train True $@  

#Train
bash run_mf.sh schneider  

# We select the model with the performance on all routes in the validation dataset

#Retrosynthesis Test
python retrosynthesis_test.py -save_dir data -f_atoms data/atom_list.txt -gpu 0 -seed 42 -beam_size 10 -epoch_for_test 100  

#Retro Star Zero Search
python retro_star_0.py -save_dir data -f_atoms data/atom_list.txt -gpu 0 -seed 42 -beam_size 5 -epoch_for_search 100  

#Retro Star Search
python get_reaction_cost.py -save_dir data -f_atoms data/atom_list.txt -gpu 0 -seed 42 -beam_size 10 -epoch_for_search 100  
python get_molecule_cost.py  
python value_mlp.py  
python retro_star.py -save_dir data -f_atoms data/atom_list.txt -gpu 0 -seed 42 -beam_size 5 -epoch_for_search 100  

#Greedy DFS Search
python greedy_dfs.py -save_dir data -f_atoms data/atom_list.txt -gpu 0 -seed 42 -beam_size 5 -epoch_for_search 100  
```



## Megan
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 Megan/data/  
mv Megan/data/valid_dataset.json Megan/data/val_dataset.json  
 
cd Megan  
source env.sh  

#Data Process  
python json2csv.py  
python acquire.py uspto_50k  
python featurize.py uspto_50k megan_16_bfs_randat  

#Train
python bin/train.py uspto_50k models/uspto_50k  

# We select the model by the original' code's setting  

#Retrosynthesis Test
python bin/retrosynthesis_test.py models/uspto_50k --beam-size 10  

#Retro Star Search
python bin/get_reaction_cost.py models/uspto_50k --beam-size 10  
python bin/get_molecule_cost.py  
python bin/value_mlp.py  
python bin/retro_star.py models/uspto_50k --beam-size 5  

#Retro Star Zero Search
python bin/retro_star_0.py models/uspto_50k --beam-size 5  

#Greedy DFS Search
python bin/greedy_dfs.py models/uspto_50k --beam-size 5  
```



## GraphRetro  
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 GraphRetro/datasets/uspto-50k  
cd GraphRetro  
export SEQ_GRAPH_RETRO=$(pwd)  
python setup.py develop  

#Data Process
mv datasets/uspto-50k/valid_dataset.json datasets/uspto-50k/eval_dataset.json  
python json2csv.py  
python data_process/canonicalize_prod.py --filename train.csv  
python data_process/canonicalize_prod.py --filename eval.csv  
python data_process/canonicalize_prod.py --filename test.csv  
python data_process/parse_info.py --mode train  
python data_process/parse_info.py --mode eval  
python data_process/parse_info.py --mode test  
python data_process/core_edits/bond_edits.py  
python data_process/lg_edits/lg_classifier.py  
python data_process/lg_edits/lg_tensors.py  

#Train
python scripts/benchmarks/run_model.py --config_file configs/single_edit/defaults.yaml  
python scripts/benchmarks/run_model.py --config_file configs/lg_ind/defaults.yaml  

# We select the model by the original' code's setting  

#Retrosynthesis Test
python scripts/eval/retrosynthesis_test.py --beam_size 10 --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models  

#Retro Star Search
python scripts/eval/get_reaction_cost.py --beam_size 10 --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models  
python scripts/eval/get_molecule_cost.py  
python scripts/eval/value_mlp.py  
python scripts/eval/retro_star.py --beam_size 5 --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models  

#Retro Star Zero Search
python scripts/eval/retro_star_0.py --beam_size 5 --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models  

#Retrosynthesis Test
python scripts/eval/greedy_dfs.py --beam_size 5 --edits_exp SingleEdit_20220823_044246 --lg_exp LGIndEmbed_20220823_04432 --edits_step best_model --lg_step best_model --exp_dir models  
```



## G2Gs  
```bash
cp train_dataset.json valid_dataset.json test_dataset.json zinc_stock_17_04_20.hdf5 G2Gs/datasets/  
cd G2Gs  

#Train
python script/train.py -g [0]  

# We select the model by the original' code's setting  

#Retrosynthesis Test
python script/retrosynthesis_test.py -g [0] -k 10 -b 1  

#Retro Star Search
python script/get_reaction_cost.py -g [0] -k 10 -b 1  
python get_molecule_cost.py  
python value_mlp.py  
python script/retro_star.py -g [0] -k 5 -b 1  

#Retro Star Zero Search
python script/retro_star_0.py -g [0] -k 5 -b 1  

#Greedy DFS Search
python script/greedy_dfs.py -g [0] -k 5 -b 1  
```

## Acknowledgement  
My deepest thanks to Binghong Chen and Samuel Genheden for very helpful discussions on their benchmarks (Retro* and PaRoutes)!  

## Reference  

Retrosim: https://github.com/connorcoley/retrosim  
Neuralsym: https://github.com/linminhtoo/neuralsym  
GLN: https://github.com/Hanjun-Dai/GLN  
G2Gs: https://torchdrug.ai/docs/tutorials/retrosynthesis  
GraphRetro: https://github.com/vsomnath/graphretro  
Transformer: https://github.com/bigchem/synthesis  
Megan: https://github.com/molecule-one/megan  



## Citation
```
@inproceedings{liu2023fusionretro,
  title={FusionRetro: Molecule Representation Fusion via In-Context Learning for Retrosynthetic Planning},
  author={Liu, Songtao and Tu, Zhengkai and Xu, Minkai and Zhang, Zuobai and Lin, Lu and Ying, Rex and Tang, Jian and Zhao, Peilin and Wu, Dinghao},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```
