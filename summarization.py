from lsa_summarizer import LsaSummarizer
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

source_file = "original_text.txt"

with open(source_file, "r", encoding='utf-8') as file:
    text = file.readlines()



summarizer = LsaSummarizer()

stopwords = stopwords.words('portuguese')
summarizer.stop_words = stopwords
summary =summarizer(text[0], 3)

print("====== Original text =====")
print(text)
print("====== End of original text =====")



print("\n========= Summary =========")

print(" ".join(summary))
print("========= End of summary =========")


