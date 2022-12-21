import sys
import torch
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TokenClassificationPipeline,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    AdamW,
    get_scheduler,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from transformers import EarlyStoppingCallback
import sacrebleu
metric = datasets.load_metric("sacrebleu")
from datasets import set_caching_enabled
import pickle
import pandas as pd

from tqdm import tqdm


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(preds, labels, tokenizer_name):
    # print(preds, labels)
    result_1 = []
    # result = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
   
    # preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    metric = datasets.load_metric("sacrebleu")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    metric1= datasets.load_metric("rouge")
    result1 = metric1.compute(predictions=decoded_preds, references=decoded_labels, use_aggregator=False)
    #import pdb;pdb.set_trace()
    result = {"bleu": result["score"], "rouge1": round(result1["rouge1"][0].fmeasure, 3),  "rougeL": round(result1["rouge1"][0].fmeasure, 3)}

    return result, decoded_preds, decoded_labels


if __name__ == "__main__":   

  ap = argparse.ArgumentParser()
  ap.add_argument("-results_file", "--results_file", type = str, help="path to pickle file")
  ap.add_argument("-tokenizer_name", "--tokenizer_name", type=str, help="tokenizer name")
  ap.add_argument("-save_dir", "--save_dir", type=str)
  ap.add_argument("-model_name", "--model_name", type=str)
  args = vars(ap.parse_args())

  for ii, item in enumerate(args):
    
    print(item + ": " + str(args[item]))
  set_caching_enabled(False)

  device = "cuda"

  f = open(args['results_file'], 'rb')
  obj= pickle.load(f)
  #print(type(obj.predictions))

  bleu = []
  rouge1 = []
  rougeL = []
  data1 = obj.predictions
  data2 = obj.label_ids
  reference = []
  predictions = []
  for i in tqdm(range(len(data1))):
    #print(i)
    res, decoded_preds, decoded_labels = compute_metrics([data1[i]], [data2[i]], args['tokenizer_name'])
    #import pdb;pdb.set_trace()
    bleu.append(res['bleu'])
    rouge1.append(res['rouge1'])
    rougeL.append(res['rougeL'])
    reference.append(decoded_labels[0])
    predictions.append(decoded_preds[0])

  df = pd.DataFrame(list(zip(reference, predictions)),columns =['Reference', 'Predictions'])
  df.to_csv(args['save_dir'] + '/' + args['model_name']  +'_predictions.csv', index=False)
  print("Final scores:")
  print("Bleu: {}".format(np.mean(bleu)))
  print("Rouge1: {}".format(np.mean(rouge1) * 100))
  print("RougeL: {}".format(np.mean(rougeL) * 100))


