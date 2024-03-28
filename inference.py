from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
import os
from model_asr import Model_ASR
from model_asr import Model_ASR

async def infer(path_audio:str = '',labId:str = "video_malicious_detection", ckpt_number:int = 1, model_name:str = "google-bert/bert-base-multilingual-uncased", sample_model_dir:str = ''):
    """
    Thực hiện Infer một đoạn audio với mô hình đã chọn
    Parameters
    ----------
    path_audio : str, require, default: '' , Audio cần Infer
    labId : str, require, default: 'video_malicious_detection' , Id của bài Lab
    ckpt_number : int, require, default: 1 , Số hiệu của check point
    model_name : str, require, default: 'distilbert-base-uncased' , Tên của mô hình sử dụng để huấn luyện
    sample_model_dir : str, require, default: '' , Đường dẫn tới check-point thực hiện Infer

    """

    #Load Model
    if sample_model_dir:
        model_dir = sample_model_dir
    else:
        model_dir = f'./modelDir/{labId}/log_train/{model_name}'
      
    if sample_model_dir:
        ckpt_path =  os.path.join (model_dir, 'ckpt')
    else:
        ckpt_path =  os.path.join (model_dir, 'ckpt-'+str (ckpt_number))

    model_asr = Model_ASR()
    text = model_asr.infer(path_audio)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    classifier = pipeline(task = 'text-classification', model = model,tokenizer = tokenizer)
    text = preprocessing_text(text)
    result = classifier(text)
    return {
        'label': result[0]['label'],
        'score': result[0]['score'],
        'text': text,
        'model_checkpoint_number': ckpt_number or 'Invalid'
    }

# if __name__ == "__main__":
#     print(infer(model_name="FacebookAI/xlm-roberta-base", 
#                 path_audio = "/mnt/wsl/PHYSICALDRIVE0p1/toan/crawl_audio_youtube/audios/first_7000_audio/Truyền_hình_Ninh_Thuận/cac_lang_cham_chuan_bi_don_tet_ramuwan_nam_2024__ntv/-6502009914870174627_34816_41336.wav"))
