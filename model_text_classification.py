from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, DataCollatorWithPadding, TrainerCallback
import pandas as pd
from datasets import Dataset
from utils import compute_metrics, preprocessing_text, load_train_valid_dataset
from copy import deepcopy

class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Thực hiện đánh giá lại tập Train
        """
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
    
class Model_Text_Classification:
    def __init__(self, train_dataset:Dataset, valid_dataset: Dataset, labId:str = "video_malicious_detection", model_name:str = 'google-bert/bert-base-multilingual-uncased'):
        """
        Parameters
        ----------
        labId : str, require, default: 'video_malicious_detection' , Nhãn của bài Lab
        model_name : str, require, default: 'google-bert/bert-base-multilingual-uncased' , Tên của mô hình cần Fine-tune có thể sử dụng các mô hình có sẵn trên Hugging face khác như: vinai/phobert-base, FacebookAI/xlm-roberta-base, ...
        train_data_dir : str, require, default: './datasets/train.csv' , Đường dẫn tới file Train.csv
        val_size : float, require, default: 0.1 , Tỷ lệ tập Valid

        """
        
        self.id2label = {0: "normal", 1: "malicious"}
        self.label2id = {"normal": 0, "malicious": 1}
        self.labId = labId
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, id2label=self.id2label, label2id=self.label2id
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_dataset, self.valid_dataset = train_dataset, valid_dataset
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    
    def preprocess_function(self, examples):
        """
        Tiền xử lý dữ liệu
        """
        return self.tokenizer(examples["text"], truncation=True)
    
    def train(self, learning_rate:float = 1e-5, EPOCHS:int = 10, BS:int = 16):
        """
        Mô hình thực thi huyến luyện
        Parameters
        ----------
        learning_rate : float, require, default: 1e-05 , Learning reate
        EPOCHS : int, require, default: 10 , Số epochs huấn luyện
        BS : int, require, default: 16 ,  Độ lớn của Batch Size
        """
        self.train_dataset = self.train_dataset.map(self.preprocess_function, batched=True)
        self.valid_dataset = self.valid_dataset.map(self.preprocess_function, batched=True)
        training_args = TrainingArguments(
            output_dir=f"./modelDir/{self.labId}/log_train/{self.model_name}",
            evaluation_strategy="epoch",
            save_strategy="no",
            per_device_train_batch_size=BS,
            per_device_eval_batch_size=BS,
            learning_rate=learning_rate,
            num_train_epochs=1,
            lr_scheduler_type='constant',
            logging_strategy = "no",
            report_to=["tensorboard"],
        )
        trainer = Trainer(
            model = self.model,
            args = training_args,
            train_dataset = self.train_dataset, # type: ignore
            eval_dataset = self.valid_dataset, # type: ignore
            tokenizer = self.tokenizer,
            data_collator = self.data_collator,
            compute_metrics = compute_metrics, # type: ignore
        )
        trainer.add_callback(CustomCallback(trainer))
        for _ in range(EPOCHS):
            trainer.train()
            trainer.save_model(f"./modelDir/{self.labId}/log_train/{self.model_name}/ckpt-{_+1}")
            yield {
                "epoch" : _ + 1,
                "train_accuracy" : trainer.state.log_history[0]["train_accuracy"],
                "train_f1_score": trainer.state.log_history[0]["train_f1_score"],
                "train_loss": trainer.state.log_history[0]["train_loss"],
                "eval_accuracy" : trainer.state.log_history[1]["eval_accuracy"],
                "eval_loss": trainer.state.log_history[1]["eval_loss"],
                "eval_f1_score": trainer.state.log_history[1]["eval_f1_score"]
            }
            
# if __name__ == "__main__":
#     model = Model_Malicious_Detection(model_name="FacebookAI/xlm-roberta-base")
#     results = []
#     results_generator = model.train(EPOCHS=10)
#     for result in results_generator:
#         results.append(result)
#     print(results)
