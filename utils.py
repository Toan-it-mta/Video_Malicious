import evaluate
import numpy as np
import re
from sklearn.metrics import f1_score
from datasets import Dataset
import pandas as pd
import os
from model_asr import Model_ASR
import torch 
import gc

accuracy = evaluate.load("accuracy")

def compute_metrics_acc_score(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def compute_metrics_f_score(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    score = f1_score(labels,predictions, average='macro')
    return {"f1_score": float(score)}

def compute_metrics(eval_pred):
    # Tính toán và trả về độ đo hiệu suất từ nhiều hàm tính toán độ đo riêng lẻ
    metric_results = {}
    for metric_function in [compute_metrics_acc_score, compute_metrics_f_score]:
        metric_results.update(metric_function(eval_pred))
    return metric_results

def preprocessing_text(text):
    # emoji_pattern = re.compile("["
    #             u"\U0001F600-\U0001F64F"  # emoticons
    #             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    #             u"\U0001F680-\U0001F6FF"  # transport & map symbols
    #             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    #             u"\U00002702-\U000027B0"
    #             u"\U000024C2-\U0001F251"
    #             u"\U0001f926-\U0001f937"
    #             u'\U00010000-\U0010ffff'
    #             u"\u200d"
    #             u"\u2640-\u2642"
    #             u"\u2600-\u2B55"
    #             u"\u23cf"
    #             u"\u23e9"
    #             u"\u231a"
    #             u"\u3030"
    #             u"\ufe0f"
    # "]+", flags=re.UNICODE)
    # text = emoji_pattern.sub(r'', text)
    text = re.sub("\r", "\n", text)
    text = re.sub("\n{2,}", "\n", text)
    text = re.sub("…", ".", text)
    text = re.sub("/.{2,}", ".", text)
    text = text.strip()
    text = text.lower()
    return text

def load_train_valid_dataset(path_train_data:str, val_size:float):
    """
    Load bộ dữ liệu và chia train/valid
    Parameters
    ----------
    path_train_data : str , Đường dẫn tới file train.csv
    val_size : float , Tỷ lệ của tập Valid
    """
    df = pd.read_csv(path_train_data)
    del df['path']
    df = df[df['text'].notna()]
    df_valid = df.sample(frac=val_size)
    df_valid['text'] = df_valid['text'].apply(preprocessing_text)
    df_train = df.drop(df_valid.index)
    df_train['text'] = df_train['text'].apply(preprocessing_text)
    train_dataset = Dataset.from_pandas(df_train)
    valid_dataset = Dataset.from_pandas(df_valid)
    return train_dataset, valid_dataset

# def convert_mp4_to_mp3(path_folder_mp4:str, path_folder_mp3:str):
#     for file in os.listdir(path_folder_mp4):
#                 try : 
#                     if re.search('mp4', file):
#                         mp4_path = os.path.join(path_folder_mp4,file)
#                         mp3_path = os.path.join(path_folder_mp3,os.path.splitext(file)[0]+'.mp3')
#                         new_file = mp.AudioFileClip(mp4_path)
#                         new_file.write_audiofile(mp3_path)
#                 except:
#                     continue
                
def get_text_from_file_mp3(path_file_mp4:str, model_asr: Model_ASR):
    # path_file_mp3 = os.path.splitext(path_file_mp4)[0]+'.mp3'
    # file_mp3 = mp.AudioFileClip(path_file_mp4)
    # file_mp3.write_audiofile(path_file_mp3)
    text = model_asr.infer(os.path.join("./datasets",path_file_mp4))
    # os.remove(path_file_mp3)
    return text

def processing_dataset(path_dataset_csv:str):
    df = pd.read_csv(path_dataset_csv)
    if 'text' not in df:
        model_asr = Model_ASR()
        df['text'] = df['path'].apply(lambda x: get_text_from_file_mp3(x, model_asr))
        df.to_csv(path_dataset_csv, index=False)
        model_asr.model.to('cpu')
        del model_asr
        torch.cuda.empty_cache()
        gc.collect()	

    