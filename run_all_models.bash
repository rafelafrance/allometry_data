#!/usr/bin/env bash

./sheet_dissection.py @args/sheet_dissection.args

./run_model.py @args/run_model_1.args
./run_model.py @args/run_model_2.args
./run_model.py @args/run_model_3.args
./run_model.py @args/run_model_4.args
./run_model.py @args/run_model_5.args
./run_model.py @args/run_model_6.args
./run_model.py @args/run_model_7.args

./ensemble.py @args/ensemble.args
