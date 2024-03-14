import evaluate
import numpy as np
import re
from sklearn.metrics import f1_score

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
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
    "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub("\r", "\n", text)
    text = re.sub("\n{2,}", "\n", text)
    text = re.sub("…", ".", text)
    text = re.sub("\.{2,}", ".", text)
    text = text.strip()
    text = text.lower()
    return text
