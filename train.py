#!/usr/bin/env python3

from simpletransformers.classification import ClassificationModel
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

model_args = dict( num_train_epochs=5 )

model = ClassificationModel("bert",
                            "prajjwal1/bert-tiny",
                            args=model_args)

train_df = feather.read_feather("train.arrow")
eval_df =  feather.read_feather("test.arrow")
model.train_model(train_df,eval_df=eval_df)
