import argparse
import pandas as pd
import pickle




def add_human_gen_cs(df_train, train_pickle, new_train):

    code_switched_human = []
    for i in range(len(df_train)):
        if df_train.iloc[i]['English'] in train_pickle.keys():
            code_switched_human.append(train_pickle.get(df_train.iloc[i]['English'])[0])

    new_train['target'] = code_switched_human
    new_train = new_train.replace('\n', '', regex=True)
    new_train["source"] = new_train[["English", "Hindi"]].apply(" ".join, axis=1)
    new_train = new_train.drop(['English', 'Hindi'], axis=1)
    return new_train


def format_dataset(csv_path, pickle_path, save_dir):
    df_train = pd.read_csv(csv_path + 'train.csv')
    df_val = pd.read_csv(csv_path + 'valid.csv')
    df_test = pd.read_csv(csv_path + 'test.csv')
    print("Synthetic dataset:")
    print(f'length of train: {len(df_train)}')
    print(f'length of valid: {len(df_val)}')
    print(f'length of test: {len(df_test)}')

    f = open(pickle_path + 'train_human_generated.pkl', 'rb')
    train_pickle = pickle.load(f)
    f = open(pickle_path + 'valid_human_generated.pkl', 'rb')
    val_pickle = pickle.load(f)
    f = open(pickle_path + 'test_human_generated.pkl', 'rb')
    test_pickle = pickle.load(f)
    print("Human generated dataset:")
    print(f'length of train: {len(train_pickle)}')
    print(f'length of valid: {len(val_pickle)}')
    print(f'length of test: {len(test_pickle)}')

    new_train = df_train[['English', 'Hindi']].copy()
    new_val = df_val[['English', 'Hindi']].copy()
    new_test = df_test[['English', 'Hindi']].copy()

    print("Formatted datasets:")
    train = add_human_gen_cs(df_train, train_pickle, new_train)
    val = add_human_gen_cs(df_val, val_pickle, new_val)
    test = add_human_gen_cs(df_test, test_pickle, new_test)

    print(f'length of train: {len(train)}')
    print(f'columns of train: {train.columns}')
    print(f'length of valid: {len(val)}')
    print(f'columns of train: {val.columns}')
    print(f'length of test: {len(test)}')
    print(f'columns of train: {test.columns}')

    train.to_csv(save_dir + 'train_df.csv', index=False)
    val.to_csv(save_dir + 'val_df.csv', index=False)
    test.to_csv(save_dir + 'test_df.csv', index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-csv_path", "--csv_path", required=True, type=str, help='output directory')
    ap.add_argument("-pickle_path", "--pickle_path", required=True, type=str, help='output directory')
    ap.add_argument("-save_dir", "--save_dir", required=True, type=str, help='output directory')

    args = vars(ap.parse_args())
    for ii, item in enumerate(args):
        print(item + ': ' + str(args[item]))
        
    format_dataset(args['csv_path'], args['pickle_path'], args['save_dir'])