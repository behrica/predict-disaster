#!/usr/bin/env python3

from simpletransformers.classification import ClassificationModel
import pandas as pd
import pyarrow.feather as feather
import yaml
import json

with open("params.yaml", 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


model_args =  {"use_cuda": True,
               "evaluate_during_training_steps": 100,
               "evaluate_during_training": True,
               "evaluate_during_training_verbose": True} | params['train']


train_df = feather.read_feather("train.arrow")
eval_df =  feather.read_feather("test.arrow")


model = ClassificationModel(params['train']['model_type'],
                            params['train']['model_name'],
                            args=model_args)



model.train_model(train_df,eval_df=eval_df)

eval_result =  model.eval_model(eval_df)


with open('eval.json', 'w') as outfile:
    json.dump(
        {"train":
         {"mcc": eval_result[0]['mcc']}},
        outfile)
