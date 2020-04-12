This code implements the summarization of text documents using Latent Semantic Analysis. For a good starting point to the LSA models in summarization, check this [paper](https://www.researchgate.net/publication/220195824_Text_summarization_using_Latent_Semantic_Analysis) and [this one](http://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf).

# Running this code

Firstly, It is necessary to download 'punkts' and 'stopwords' from nltk data. For that, run the code:

```python
import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
```

Further, run ```python summarization.py```

# Requirements

* Python 3.x
* numpy
* NLTK