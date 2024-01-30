#!/bin/bash

echo "Running $file..."
python ./run_train_model.py --model almt --epochs 15 --eval_per_epoch
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model mult --epochs 15 --eval_per_epoch
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model lmf --epochs 15 --eval_per_epoch
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model base --epochs 15 --eval_per_epoch
echo "----------------------------------------"

echo "All scripts have been executed."

echo "Running $file..."
python ./run_train_model.py --model almt --epochs 15 --eval_per_epoch --modal i
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model mult --epochs 15 --eval_per_epoch --modal i
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model lmf --epochs 15 --eval_per_epoch --modal i
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model base --epochs 15 --eval_per_epoch --modal i
echo "----------------------------------------"

echo "All scripts(image only) have been executed."


echo "Running $file..."
python ./run_train_model.py --model almt --epochs 15 --eval_per_epoch --modal t
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model mult --epochs 15 --eval_per_epoch --modal t
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model lmf --epochs 15 --eval_per_epoch --modal t
echo "----------------------------------------"

echo "Running $file..."
python ./run_train_model.py --model base --epochs 15 --eval_per_epoch --modal t
echo "----------------------------------------"

echo "All scripts(text only) have been executed."