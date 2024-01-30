#!/bin/bash

echo "Running $file..."
python ./run_train_model.py --model almt --epochs 10 --eval_per_epoch
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model mult --epochs 10 --eval_per_epoch
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model lmf --epochs 15 --eval_per_epoch
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model base --epochs 7 --eval_per_epoch
echo "----------------------------------------"

echo "All scripts have been executed."