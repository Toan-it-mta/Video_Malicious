from model import Model_Malicious_Detection

def train(labId:str = "malicious_detection", model_name:str = 'google-bert/bert-base-multilingual-uncased', train_data_dir:str = "./datasets/train.csv", val_size:float = 0.2,
                learning_rate:float = 1e-5, epochs:int = 3, batch_size:int = 16):
    """
    Parameters
    ----------
    labId : str, require, default: 'malicious_detection' , Nhãn của bài Lab
    model_name : str, require, default: 'google-bert/bert-base-multilingual-uncased' , Tên của mô hình cần Fine-tune có thể sử dụng các mô hình có sẵn trên Hugging face khác như: vinai/phobert-base, FacebookAI/xlm-roberta-base, ...
    train_data_dir : str, require, default: './datasets/train.csv' , Đường dẫn tới file Train.csv
    val_size : float, require, default: 0.1 , Tỷ lệ tập Valid
    learning_rate : float, require, default: 1e-05 , Learning reate
    epochs : int, require, default: 3 , Số lượng epochs cần huấn luyện
    batch_size : int, require, default: 16 , Độ lớn của Batch Size

    """
    
    model = Model_Malicious_Detection(labId=labId, model_name=model_name, train_data_dir=train_data_dir,val_size=val_size)
    train_output = model.train(learning_rate=learning_rate,EPOCHS=epochs, BS=batch_size)
    for res_per_epoch in train_output:
	    yield res_per_epoch
    
if __name__ == '__main__':
    respones = train(model_name="FacebookAI/xlm-roberta-base", epochs=10)
    for res in respones:
        print(res)
