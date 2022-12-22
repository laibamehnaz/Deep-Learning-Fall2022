# Deep Learning 2022 Final project

### Dataset:
For our project we have used the [HINGE](https://aclanthology.org/2021.eval4nlp-1.20.pdf) dataset. The HINGE dataset consists of parallel English and Hindi sentences, and their associated Hindi-English codeswitched translation.

To run our code use the following commands:

#### Training the model on mT5
```
python DLfinalProject.py --output_dir /output/ 
    --train_data_dir /train_df.csv 
    --test_data_dir /test_df.csv 
    --val_data_dir /val_df.csv 
    --model_name google/mt5-small 
    --epochs 15 
    --source_lang source 
    --target_lang target
```
#### Training the model on mBART
```
python DLfinalProject.py --output_dir /output/ 
    --train_data_dir /train_df.csv 
    --test_data_dir /test_df.csv 
    --val_data_dir /val_df.csv 
    --model_name facebook/mbart-large-cc25
    --epochs 15 
    --source_lang source 
    --target_lang target
```
Note: The above command uses mBART, however, in our paper we have used student mBART. If you wish to use student mBART, please contact us on lm4428@nyu.edu or gg2612@nyu.edu. 


To evaluate the generated predictions use the following code:

### To evaluate generated predictions for mT5
```
python evaluateDLproject.py --results_file /pickles/test_results_mt5.pickle 
    --tokenizer_name google/mt5-small
    --save_dir /output/
    --model_name mt5
```

### To evaluate generated predictions for mBART
```
python evaluateDLproject.py --results_file /pickles/test_results_mBART.pickle 
    --tokenizer_name facebook/mbart-large-cc25
    --save_dir /output/
    --model_name mBART
```
