from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, TrainerCallback
import pandas as pd
from datasets import Dataset
from utils import compute_metrics, preprocessing_text, processing_dataset
import os
from model_asr import Model_ASR

async def test(path_test_data:str = './datasets/test.csv', labId:str = "malicious_detection", ckpt_number:int = 1, model_name:str = "google-bert/bert-base-multilingual-uncased", sample_model_dir:str = ''):
    """
    Thực hiện Test mô hình
    Parameters
    ----------
    test_data_dir : str, optional, default: './datasets/test.csv' , Đường dẫn tới file chứa tập test (test.csv)
    labId : str, optional, default: 'malicious_detection' , Id của bài Lab
    ckpt_number : int, optional, default: 1 , Số hiệu của check point
    model_name : str, require, default: 'google-bert/bert-base-multilingual-uncased' , Tên của mô hình cần Fine-tune có thể sử dụng các mô hình có sẵn trên Hugging face khác như: vinai/phobert-base, FacebookAI/xlm-roberta-base, ...
    sample_model_dir : str, require, default: '' , Đường dẫn tới check-point thực hiện Infer
    """
    model_asr = Model_ASR()
    processing_dataset(path_test_data, model_asr)
    # Load test dataset
    df = pd.read_csv(path_test_data)
    df = df[df['text'].notna()]
    df['text'] = df['text'].apply(preprocessing_text)
    del df['path']
    test_dataset = Dataset.from_pandas(df)

    #Load Model
    if sample_model_dir:
        model_dir = sample_model_dir
    else:
        model_dir = f'./modelDir/{labId}/log_train/{model_name}'
      
    if sample_model_dir:
        ckpt_path =  os.path.join (model_dir, 'ckpt')
    else:
        ckpt_path =  os.path.join (model_dir, 'ckpt-'+str (ckpt_number))
      
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    test_dataset = test_dataset.map(preprocess_function, batched=True)
    trainer = Trainer(
        model = model, # type: ignore
        eval_dataset = test_dataset, # type: ignore
        tokenizer = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_metrics, # type: ignore
    )
    result = trainer.evaluate()
    return {
        'test_acc': result['eval_accuracy'],
        'test_f1_score': result['eval_f1_score'],
        'test_loss': result["eval_loss"],
        'model_checkpoint_number': ckpt_number or "Invalid"
    }

# if __name__ == "__main__":
#     for i in range(10):
#         idx = i+1
#         print(test(model_name="FacebookAI/xlm-roberta-base", ckpt_number=idx))
