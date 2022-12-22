import argparse
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np
from transformers import EarlyStoppingCallback
import sacrebleu
import evaluate

metric = evaluate.load("sacrebleu")
from datasets import set_caching_enabled

set_caching_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("-output_dir", "--output_dir", required=True, type=str, help='output directory')
ap.add_argument("-train_data_dir", "--train_data_dir", required=True, type=str, help='output directory')
ap.add_argument("-test_data_dir", "--test_data_dir", required=True, type=str, help='output directory')
ap.add_argument("-val_data_dir", "--val_data_dir", required=True, type=str, help='output directory')
ap.add_argument("-model_name", "--model_name", required=True, type=str, help=' model_name')
ap.add_argument("-epochs", "--epochs", required=True, type=int, help='epochs')
ap.add_argument("-source_lang", "--source_lang", required=True, type=str, help=' source_lang')
ap.add_argument("-target_lang", "--target_lang", required=True, type=str, help=' source_lang')

args = vars(ap.parse_args())
for ii, item in enumerate(args):
    print(item + ': ' + str(args[item]))

data_files = {"train": args['train_data_dir'], "validation": args['val_data_dir'], "test": args['test_data_dir']}
raw_datasets = load_dataset('csv', data_files=data_files)
raw_datasets.cleanup_cache_files()
# import pdb; pdb.set_trace()

model_name = args['model_name']  # you can specify the model size here
tokenizer = AutoTokenizer.from_pretrained(model_name)

if "mbart" in model_name:
    if args['source_lang'] == "hi":
        tokenizer.src_lang = "hi-IN"
    else:
        tokenizer.src_lang = "hi-IN"

    if args['target_lang'] == "en":
        tokenizer.tgt_lang = "en-XX"
    else:
        tokenizer.tgt_lang = "en-XX"

if model_name in ["google/mt5-base", "/content/drive/MyDrive/Deep-Learning-Fall2022-main/output/checkpoint-2580/", "/content/drive/MyDrive/Deep-Learning-Fall2022-main (1)/Deep-Learning-Fall2022-main/output/checkpoint-1634/"]:
    prefix = "translate Hindi to English: "
else:
    prefix = ""

max_input_length = 128
max_target_length = 128
source_lang = args['source_lang'] #source
target_lang = args['target_lang'] #target


def preprocess_function(examples):
    inputs = [prefix + ex for ex in examples[source_lang]]
    targets = [ex for ex in examples[target_lang]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True,
                             return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True,
                           return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, load_from_cache_file=False,
                                      remove_columns=["source", "target"])
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

batch_size = 2
model_name = model_name.split("/")[-1]
train_args = Seq2SeqTrainingArguments(output_dir=args['output_dir'],
                                      # do_train=True,
                                      logging_first_step=True,
                                      logging_strategy="steps",
                                      logging_steps=10,
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",
                                      gradient_accumulation_steps=16,
                                      learning_rate=2e-5,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      weight_decay=0.01,
                                      save_total_limit=3,
                                      num_train_epochs=args['epochs'],
                                      predict_with_generate=True,
                                      push_to_hub=False,
                                      load_best_model_at_end=True,
                                      adafactor=True
                                      )

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    # print("decoded_preds")
    # print(decoded_preds[:10])
    # print("decoded_labels")
    # print(decoded_labels[:10])
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


trainer = Seq2SeqTrainer(
    model,
    train_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    compute_metrics=compute_metrics
)

trainer.train()

test_results = trainer.predict(tokenized_datasets["test"])

import pickle

with open(args['output_dir'] + "/test_results.pickle", "wb") as f:
    pickle.dump(test_results, f)