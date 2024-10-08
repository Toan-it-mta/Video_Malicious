from model_text_classification import Model_Text_Classification
from utils import processing_dataset, load_train_valid_dataset
import pandas as pd
import os

async def train(labId:str = "video_malicious_detection", model_name:str = 'google-bert/bert-base-multilingual-uncased', path_train_data:str = "./datasets/train.csv", val_size:float = 0.1,
                learning_rate:float = 2e-4, epochs:int = 3, batch_size:int = 8):
    """
    Parameters
    ----------
    labId : str, require, default: 'video_malicious_detection' , Nhãn của bài Lab
    model_name : str, require, default: 'google-bert/bert-base-multilingual-uncased' , Tên của mô hình cần Fine-tune có thể sử dụng các mô hình có sẵn trên Hugging face khác như: vinai/phobert-base, FacebookAI/xlm-roberta-base, ...
    train_data_dir : str, require, default: './datasets/train.csv' , Đường dẫn tới file Train.csv
    val_size : float, require, default: 0.1 , Tỷ lệ tập Valid
    learning_rate : float, require, default: 1e-05 , Learning reate
    epochs : int, require, default: 3 , Số lượng epochs cần huấn luyện
    batch_size : int, require, default: 16 , Độ lớn của Batch Size

    """
    processing_dataset(path_train_data)
    train_dataset, valid_dataset = load_train_valid_dataset(path_train_data, val_size)
    model_text_classification = Model_Text_Classification(labId=labId, model_name=model_name, train_dataset=train_dataset, valid_dataset=valid_dataset)
    train_output = model_text_classification.train(learning_rate=learning_rate,EPOCHS=epochs, BS=batch_size)

    
    for res_per_epoch in train_output:
	    yield res_per_epoch
         
# if __name__ == '__main__':
#     respones = train(model_name="vinai/phobert-base", epochs=3)
#     for res in respones:
#         print(res)
