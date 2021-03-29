#!/usr/bin/env bash

./02_generate_images.py @args/02_generate_train_images.args --seed=4512 --text-dir ./data/page/pool1 &
./02_generate_images.py @args/02_generate_train_images.args --seed=9976 --text-dir ./data/page/pool2 &
./02_generate_images.py @args/02_generate_train_images.args --seed=3048 --text-dir ./data/page/pool3 &
./02_generate_images.py @args/02_generate_train_images.args --seed=7712 --text-dir ./data/page/pool4 &
./02_generate_images.py @args/02_generate_valid_images.args --text-dir ./data/page/pool_valid &
./02_generate_images.py @args/02_generate_test_images.args --text-dir ./data/page/pool_test &
