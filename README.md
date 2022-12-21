# Deep-Learning-Fall2022
Final project for Deep Learning Fall2022

to run the code:

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
            
```
python evaluateDLproject.py --results_file /test_results_mt5.pickle 
    --tokenizer_name google/mt5-small
    --save_dir /output
    --model_name mt5
```
