from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
from operator import itemgetter

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Model_ASR:
    def __init__(self, model_name='./modelDir/asr_models/PhoWhisper-medium') -> None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
    def infer(self,audiopath:str) -> str:
        pipe = pipeline(
        "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=20,
            batch_size=1,
            return_timestamps=True,
            torch_dtype=self.torch_dtype,
            device=device,
            generate_kwargs={"task": "transcribe","language":'vi'}
        )
        prediction = pipe(audiopath)
        result_string = " ".join(map(itemgetter('text'), prediction["chunks"]))
        return result_string

# if __name__ == "__main__":
#     whisper = Whisper_Model()
#     result = whisper.infer("/mnt/wsl/PHYSICALDRIVE0p1/toan/crawl_audio_youtube/sau_day_se_la_huong_dan_su_dun_100723032428.wav")
