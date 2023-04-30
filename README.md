# CS542-Competition

Before first execution:

```
conda create -n cs542 -c conda-forge -c huggingface scikit-learn numpy transformers datasets
conda activate cs542
pip install evaluate
pip install tensorboardX
```

Before every execution:

```
conda activate cs542
```

To download data in project directory (less than 1GB in total, install gdown before download):

```
gdown download 16TI8uVk3No1HCx765PrSWLDMmnnqmqec
gdown download 1RDgFbMApfv8lkXi7PVeBoqkHK9xwItxs
```

To finetune a model with boolean answers only:

```
python bool_extraction.py

python run_lm_finetuning.py --output_dir=output --model_type=gpt2  --model_name_or_path=gpt2  --do_train --train_data_file=datasets/bool_training.json --do_eval --eval_data_file=datasets/bool_testing.json --per_gpu_eval_batch_size 2 --per_gpu_train_batch_size 2
```

To reproduce the submited output:

```
python bool_extraction.py

python run_lm_finetuning.py --output_dir=output --model_type=gpt2  --model_name_or_path=gpt2  --do_train --train_data_file=datasets/bool_training.json --do_eval --eval_data_file=datasets/bool_testing.json --per_gpu_eval_batch_size 2 --per_gpu_train_batch_size 2

python inference.py
```

Pack the output predictions.pkl in `submission.zip`, then put it in the same directory with the provided `evaluation.ipynb`, click "run all" or execute all code block in the notebook, the output will be:

```
T/F: 49.39, MCQ: 39.13, NUM: 23.28
Combined Metric: 111.80
```
