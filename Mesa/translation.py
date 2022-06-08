import googletrans
from googletrans import Translator

class TranslateClass:
    translated_text=''

    def __init__(self,t):
        self.translated_text=t
print("In the file")    
def translatemethod(content):
        translator = Translator()
        translated_text = translator.translate(content, src='en', dest='hi')
        print(translated_text.text)
        return translated_text.text
