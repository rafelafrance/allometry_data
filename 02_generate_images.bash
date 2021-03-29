#!/usr/bin/env bash

./02_generate_images.py @args/02_generate_train_images.args --seed=4512 --text-dir=./data/page/pool_train_1 &
./02_generate_images.py @args/02_generate_train_images.args --seed=9976 --text-dir=./data/page/pool_train_2 &
./02_generate_images.py @args/02_generate_train_images.args --seed=2048 --text-dir=./data/page/pool_train_3 &
./02_generate_images.py @args/02_generate_train_images.args --seed=7712 --text-dir=./data/page/pool_train_4 &
./02_generate_images.py @args/02_generate_valid_images.args &
./02_generate_images.py @args/02_generate_test_images.args &
