#!/usr/bin/env python3

from simpletransformers.classification import ClassificationModel
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import yaml
import json

with open("params.yaml", 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


model_args = {"use_cuda": True} | params

model = ClassificationModel("bert",
                            "prajjwal1/bert-tiny",
                            args=model_args)

train_df = feather.read_feather("train.arrow")
eval_df =  feather.read_feather("test.arrow")
model.train_model(train_df,eval_df=eval_df)

eval_result =  model.eval_model(eval_df)


with open('eval.json', 'w') as outfile:
    json.dump(
        {"train":
         {"mcc": eval_result[0]['mcc']}},
        outfile)
