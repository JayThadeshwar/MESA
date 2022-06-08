from transformers import pipeline
#  ## Setting to use the 0th GPU
# #os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class SummarizeClass:
    summarized_text=''

    def __init__(self,s):
        self.summarized_text=s

def summarizemethod(content):
    summary_text = summarizer(content, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
    print("Length of input:",len(content))
    print("Length of summary:",len(summary_text))
    return summary_text