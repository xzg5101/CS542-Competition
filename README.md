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

To download data in project directory (less than 150GB in total, install gdown before download):

```
gdown 16TI8uVk3No1HCx765PrSWLDMmnnqmqec
gdown 1RDgFbMApfv8lkXi7PVeBoqkHK9xwItxs
```
Notice this dataset is just the dataset provided by the competition.

To finetune a model with T/F questions, move to the repository directory, run:

```
python bool_extraction.py

python run_lm_finetuning_auth.py \
    --output_dir=bool_fine_tuning4 \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=datasets/bool_training.json \
    --do_eval \
    --eval_data_file=datasets/bool_testing.json \
    --num_train_epochs 5 \
    --save_steps 150 \
    --per_gpu_eval_batch_size 2 \
    --per_gpu_train_batch_size 2
```
To finetune a model with T/F questions, move to the repository directory, run:

```
python run_lm_finetuning_auth.py \
    --output_dir=choice_fine_tuning \
    --model_type=gpt2 \
    --model_name_or_path=gpt2 \
    --do_train \
    --train_data_file=datasets/choice_training.json \
    --do_eval \
    --eval_data_file=datasets/choice_testing.json \
    --save_steps 150 \
    --per_gpu_eval_batch_size 2 \
    --per_gpu_train_batch_size 2
```
To make predictions use the two models above:

```
python inference.py
'''
By excecuting the `inference.py`, the prediction will be put in `submission` direction. Pack the output predictions.pkl in `submission.zip`, then put it in the same directory with the provided `evaluation.ipynb`, click "run all" or execute all code blocks in the notebook, the output will be:

```
T/F: 24.86, MCQ: 39.19, NUM: 23.28
Combined Metric: 87.34
```

Which is better than the random baseline.
