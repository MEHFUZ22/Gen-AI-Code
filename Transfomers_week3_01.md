#Sentiment Analysis


```python
from transformers import pipeline


classifier=pipeline("sentiment-analysis")

review=[
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]


result=classifier(review)
print(result)
```

    No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Device set to use cpu
    

    [{'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
    


```python

```


```python
from transformers import pipeline


classifier=pipeline(task="sentiment-analysis")

review="The film was kind of good but not very good",




result=classifier(review)
print(result)
```

    No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Device set to use cpu
    

    {'label': 'NEGATIVE', 'score': 0.991393506526947}
    


```python
from transformers import pipeline


classifier=pipeline(task="sentiment-analysis")

review="I am good not that good",




result=classifier(review)
print(result)
```

    No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Device set to use cpu
    

    {'label': 'NEGATIVE', 'score': 0.9913764595985413}
    


```python
from transformers import pipeline


classifier=pipeline("sentiment-analysis")

reviews=[
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]


results=classifier(reviews)
print(results)
```

    No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Device set to use cpu
    

    [{'label': 'POSITIVE', 'score': 0.9598049521446228}, {'label': 'NEGATIVE', 'score': 0.9994558691978455}]
    


```python
for review ,result in zip(reviews,results):
  print(f" Review : {review} | {result['label']} | {result['score']}")
```

     Review : I've been waiting for a HuggingFace course my whole life. | POSITIVE | 0.9598049521446228
     Review : I hate this so much! | NEGATIVE | 0.9994558691978455
    

#Summarisation






```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
output_text = summary[0]['summary_text']
print(output_text)
print("Token count:", len(tokenizer.encode(output_text)))

```


```python
text = """Artificial Intelligence (AI) has rapidly become an integral part of our daily
lives. From voice assistants like Siri and Alexa to recommendation systems on
platforms like Netflix and Amazon, AI is shaping how we interact with technology.
It is no longer confined to the realm of science fiction but is now a practical tool
that enhances efficiency and convenience.
One of the key drivers of AI adoption is its ability to analyze vast amounts of data
and make predictions or decisions faster than humans. In industries like
healthcare, AI-powered systems are used to detect diseases such as cancer in
their early stages, saving countless lives. Similarly, in the financial sector, AI
algorithms help identify fraudulent transactions and streamline investment
strategies.
However, the rise of AI also raises ethical concerns. Issues such as data privacy,
algorithmic bias, and job displacement are frequently debated. While AI can
improve productivity, it also has the potential to replace human jobs, particularly
in repetitive or manual tasks. This highlights the need for responsible AI
development and robust regulations to ensure that technology benefits everyone.
Despite these challenges, the potential of AI is immense. Researchers are
constantly working on advancements in natural language processing, computer
vision, and machine learning to make AI even more intelligent and adaptable. As
technology continues to evolve, it is likely that AI will play an even greater role in
shaping our future."""

```


```python
from transformers import pipeline

summarizer=pipeline("summarization")
summary=summarizer(text)
print(summary)
```

    No model was supplied, defaulted to sshleifer/distilbart-cnn-12-6 and revision a4f8f3e (https://huggingface.co/sshleifer/distilbart-cnn-12-6).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    config.json: 0.00B [00:00, ?B/s]



    pytorch_model.bin:   0%|          | 0.00/1.22G [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/1.22G [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]


    Device set to use cpu
    

    [{'summary_text': ' Artificial Intelligence (AI) has rapidly become an integral part of our daily lives . From voice assistants like Siri and Alexa to recommendation systems on platforms like Netflix and Amazon, AI is shaping how we interact with technology . The rise of AI also raises ethical concerns over data privacy and job displacement .'}]
    


```python
print(summary[0]['summary_text'])
```

     Artificial Intelligence (AI) has rapidly become an integral part of our daily lives . From voice assistants like Siri and Alexa to recommendation systems on platforms like Netflix and Amazon, AI is shaping how we interact with technology . The rise of AI also raises ethical concerns over data privacy and job displacement .
    


```python
text = """Artificial Intelligence (AI) has rapidly become an integral part of our daily
lives. From voice assistants like Siri and Alexa to recommendation systems on
platforms like Netflix and Amazon, AI is shaping how we interact with technology.
It is no longer confined to the realm of science fiction but is now a practical tool
that enhances efficiency and convenience.
One of the key drivers of AI adoption is its ability to analyze vast amounts of data
and make predictions or decisions faster than humans. In industries like
healthcare, AI-powered systems are used to detect diseases such as cancer in
their early stages, saving countless lives. Similarly, in the financial sector, AI
algorithms help identify fraudulent transactions and streamline investment
strategies.
However, the rise of AI also raises ethical concerns. Issues such as data privacy,
algorithmic bias, and job displacement are frequently debated. While AI can
improve productivity, it also has the potential to replace human jobs, particularly
in repetitive or manual tasks. This highlights the need for responsible AI
development and robust regulations to ensure that technology benefits everyone.
Despite these challenges, the potential of AI is immense. Researchers are
constantly working on advancements in natural language processing, computer
vision, and machine learning to make AI even more intelligent and adaptable. As
technology continues to evolve, it is likely that AI will play an even greater role in
shaping our future."""

```


```python
from transformers import pipeline

summarizer=pipeline(task="summarization",model="t5-small",min_length=10,max_length=20)
summary=summarizer(text)
print(summary)
```


    config.json:   0%|          | 0.00/1.21k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/242M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/147 [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/2.32k [00:00<?, ?B/s]



    spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.39M [00:00<?, ?B/s]


    Device set to use cpu
    

    [{'summary_text': 'AI is now a practical tool that enhances efficiency and convenience . it can analyze vast amounts of data and make predictions or decisions faster than humans . in industries like healthcare, AI-powered systems are used to detect diseases .'}]
    


```python
from transformers import pipeline

summarizer=pipeline(task="summarization",model="t5-small",min_length=5,max_length=5)
summary=summarizer(text)
print(summary)
```

    Device set to use cpu
    

    [{'summary_text': 'AI is now a practical tool that enhances efficiency and convenience . it can analyze vast amounts of data and make predictions or decisions faster than humans . in industries like healthcare, AI-powered systems are used to detect diseases .'}]
    

to know the tokenizer count or lenght of output


```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")
output_text = summary[0]['summary_text']
print(output_text)
print("Token count:", len(tokenizer.encode(output_text)))

```

    AI is now a practical tool that enhances efficiency and convenience . it can analyze vast amounts of data and make predictions or decisions faster than humans . in industries like healthcare, AI-powered systems are used to detect diseases .
    Token count: 49
    


```python
from transformers import pipeline

summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")
summary = summarizer(text, min_length=30, max_length=130)
print(summary[0]['summary_text'])

```


    config.json: 0.00B [00:00, ?B/s]



    model.safetensors:   0%|          | 0.00/1.63G [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/363 [00:00<?, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]


    Device set to use cpu
    

    Artificial Intelligence (AI) has rapidly become an integral part of our daily lives. From voice assistants like Siri and Alexa to recommendation systems on Netflix and Amazon, AI is shaping how we interact with technology. But the rise of AI also raises ethical concerns.
    


```python
from transformers import pipeline

summarizer = pipeline(task="summarization", model="facebook/bart-large-cnn")
summary = summarizer(text, min_length=10, max_length=10)
print(summary[0]['summary_text'])

```

    Device set to use cpu
    

    Artificial Intelligence (AI) has
    


```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

```

    Total parameters: 406,290,432
    

#Questions and Answer


```python
from transformers import pipeline


question_answering=pipeline(task="question-answering")

context="""
Name: Mehfuz Ur Rahman
Role: Big Data Developer
Phone: +91 9886746347
Email: mehfuz0831@gmail.com
LinkedIn: linkedin.com/in/mehfuzurrahman
Location: Bangalore, India

SUMMARY:
Skilled Big Data Engineer with 3 years of proven experience in designing and implementing efficient ETL processes. Proficient in Python, SQL, PySpark, and cloud-based data platforms. Strong collaborator with cross-functional teams to deliver high-quality solutions.

SKILLS:
- Programming: Python, SQL, PySpark
- Data & Technologies: Apache Spark, Databricks, Azure Cloud, Azure Data Lake, Azure Data Factory, Azure Synapse, Hive
- Data Engineering: ETL, Data Modelling, Data Warehousing
- Tools: Git, GitHub, CI/CD
- Cloud: AWS Redshift, AWS S3
- Others: Machine Learning, Data Structures

EXPERIENCE:

**Data Engineer**
Tata Consultancy Services
Bangalore | Oct 2021 – Sep 2024

Key Responsibilities:
- Designed and implemented data pipelines using Azure Databricks and ADF, reducing data processing time by 25%.
- Applied data quality checks using SQL and PySpark, improving accuracy by 15%.
- Tuned SQL queries to optimize data retrieval by 20%.
- Enhanced logic in PySpark and SQL to handle 60% of inconsistent data, improving quality.
- Automated the MDM outbound process for seamless data integration.
- Collaborated with teams and clients, documented processes to ensure knowledge transfer.

KEY ACHIEVEMENTS:
- **TCS Higher Talent Award**: Received “Elevate Wings” badge out of 200,000 participants for exceptional performance and impact.
- **TFactor Excellence**: Achieved a T-factor score of 2.99 (benchmark: 1.99), recognized for steep learning curve and high contributions.

CERTIFICATIONS:
1. Microsoft Certified Azure Data Engineer
2. Masters Program in Data Science (Simplilearn + IBM)
3. HackerRank – SQL Advanced
4. Google Data Analytics Professional Certificate

EDUCATION:
Bachelor of Engineering – Electronics and Communication
Visvesvaraya Technological University, Bangalore
Aug 2017 – Aug 2021

LANGUAGES:
- English
- Kannada
- Hindi
- Bengali
"""
```

    No model was supplied, defaulted to distilbert/distilbert-base-cased-distilled-squad and revision 564e9b5 (https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    config.json:   0%|          | 0.00/473 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/261M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/49.0 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]


    Device set to use cpu
    

#


```python
qs="tell me the company name in which I have worked?"
result=question_answering(question=qs,context=context)
print(result)
```

    {'score': 0.42046213150024414, 'start': 825, 'end': 862, 'answer': 'Tata Consultancy Services  \nBangalore'}
    


```python
from transformers import pipeline

qa = pipeline(task="question-answering", model="deepset/roberta-base-squad2")
qs="tell me the company name in which I have worked?"

result = qa(question=qs, context=context)
print(result['answer'])

```


    config.json:   0%|          | 0.00/571 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/496M [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/79.0 [00:00<?, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    special_tokens_map.json:   0%|          | 0.00/772 [00:00<?, ?B/s]


    Device set to use cpu
    /usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py:1750: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `RobertaSdpaSelfAttention.forward`.
      return forward_call(*args, **kwargs)
    

    Microsoft Certified Azure Data Engineer
    


```python
context="""
Mehfuz Ur Rahman is a Big Data Developer based in Bangalore, India, with 3 years of experience specializing in designing and implementing efficient ETL processes. He is proficient in Python, SQL, PySpark, and various cloud platforms including Azure Databricks, Azure Data Factory, Azure Synapse, Azure Data Lake, as well as AWS services like Redshift and S3. His skills encompass Apache Spark, data modeling, data warehousing, CI/CD, and machine learning fundamentals.

From October 2021 to September 2024, Mehfuz worked as a Data Engineer at Tata Consultancy Services, where he led the development of data pipelines that reduced processing time by 25%. He improved data accuracy by 15% through quality checks and optimized SQL queries to enhance data retrieval by 20%. Mehfuz also enhanced PySpark and SQL logic to handle inconsistent data efficiently and automated master data management outbound processes. He collaborated extensively with cross-functional teams and ensured thorough documentation for smooth knowledge transfer.

Mehfuz received notable recognition including the TCS Higher Talent Award among 200,000 participants and achieved a T-factor score of 2.99, surpassing the high proficiency benchmark. His certifications include Microsoft Certified Azure Data Engineer, a Masters Program in Data Science from Simplilearn in collaboration with IBM, HackerRank SQL Advanced, and Google Data Analytics Professional Certificate.

He holds a Bachelor of Engineering degree in Electronics and Communication from Visvesvaraya Technological University, Bangalore. Mehfuz is fluent in English, Kannada, Hindi, and Bengali.
"""
```


```python
from transformers import pipeline

qa = pipeline(task="question-answering", model="deepset/roberta-base-squad2")
qs="total experience I have?"
result = qa(question=qs, context=context)
print(result['answer'])

```

    Device set to use cpu
    

    3 years
    


```python
from transformers import pipeline

qa = pipeline(task="question-answering", model="deepset/roberta-base-squad2")

def get_answer(question, context=None):
    if context is None:
        context = """
        Mehfuz Ur Rahman is a Big Data Developer based in Bangalore, India, with 3 years of experience specializing in designing and implementing efficient ETL processes. He is proficient in Python, SQL, PySpark, and various cloud platforms including Azure Databricks, Azure Data Factory, Azure Synapse, Azure Data Lake, as well as AWS services like Redshift and S3. His skills encompass Apache Spark, data modeling, data warehousing, CI/CD, and machine learning fundamentals.

        From October 2021 to September 2024, Mehfuz worked as a Data Engineer at Tata Consultancy Services, where he led the development of data pipelines that reduced processing time by 25%. He improved data accuracy by 15% through quality checks and optimized SQL queries to enhance data retrieval by 20%. Mehfuz also enhanced PySpark and SQL logic to handle inconsistent data efficiently and automated master data management outbound processes. He collaborated extensively with cross-functional teams and ensured thorough documentation for smooth knowledge transfer.

        Mehfuz received notable recognition including the TCS Higher Talent Award among 200,000 participants and achieved a T-factor score of 2.99, surpassing the high proficiency benchmark. His certifications include Microsoft Certified Azure Data Engineer, a Masters Program in Data Science from Simplilearn in collaboration with IBM, HackerRank SQL Advanced, and Google Data Analytics Professional Certificate.

        He holds a Bachelor of Engineering degree in Electronics and Communication from Visvesvaraya Technological University, Bangalore. Mehfuz is fluent in English, Kannada, Hindi, and Bengali.
        """
    result = qa(question=question, context=context)
    return result['answer']



```

    Device set to use cpu
    


```python
# Example:
print(get_answer("What are Mehfuz's achievements?"))
print(get_answer("How many years of experience does Mehfuz have?"))
```

    Microsoft Certified Azure Data Engineer
    3
    


```python
get_answer("tell me the period mehfuz worked in in tata consultancy service")
```




    'October 2021 to September 2024'



#Translation


```python
from transformers import pipeline


translator=pipeline(task="translation_en_to_fr",model='t5-small')

text='mehfuz is good guy'
result=translator(text)
print(result)
```


    config.json:   0%|          | 0.00/1.21k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/242M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/147 [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/2.32k [00:00<?, ?B/s]



    spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.39M [00:00<?, ?B/s]


    Device set to use cpu
    

    [{'translation_text': 'mehfuz est bon homme'}]
    


```python
from transformers import pipeline


translator=pipeline(task="translation_en_to_fr",model='t5-base')

text='mehfuz is good guy'
result=translator(text)
print(result)
```


    config.json:   0%|          | 0.00/1.21k [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/892M [00:00<?, ?B/s]



    generation_config.json:   0%|          | 0.00/147 [00:00<?, ?B/s]



    spiece.model:   0%|          | 0.00/792k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.39M [00:00<?, ?B/s]


    Device set to use cpu
    

    [{'translation_text': 'mehfuz est un bon homme'}]
    


```python
!cat /proc/meminfo | grep Mem
!df -h  # check disk space

```

    MemTotal:       13289424 kB
    MemFree:          873120 kB
    MemAvailable:    7887920 kB
    Filesystem      Size  Used Avail Use% Mounted on
    overlay         108G   41G   67G  38% /
    tmpfs            64M     0   64M   0% /dev
    shm             5.8G  4.0K  5.8G   1% /dev/shm
    /dev/root       2.0G  1.2G  775M  61% /usr/sbin/docker-init
    tmpfs           6.4G   20M  6.4G   1% /var/colab
    /dev/sda1        73G   42G   31G  58% /kaggle/input
    tmpfs           6.4G     0  6.4G   0% /proc/acpi
    tmpfs           6.4G     0  6.4G   0% /proc/scsi
    tmpfs           6.4G     0  6.4G   0% /sys/firmware
    

##Entities


```python
from transformers import pipeline



ner=pipeline(task='ner')

text='Elon Musk founded SpaceX in 2002 and Tesla Motors in 2003'
entities=ner(text)
print(entities)

```

    No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english and revision 4c53496 (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    


    config.json:   0%|          | 0.00/998 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/1.33G [00:00<?, ?B/s]


    Some weights of the model checkpoint at dbmdz/bert-large-cased-finetuned-conll03-english were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
    - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    


    tokenizer_config.json:   0%|          | 0.00/60.0 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]


    Device set to use cpu
    /usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py:1750: FutureWarning: `encoder_attention_mask` is deprecated and will be removed in version 4.55.0 for `BertSdpaSelfAttention.forward`.
      return forward_call(*args, **kwargs)
    

    [{'entity': 'I-PER', 'score': np.float32(0.9994467), 'index': 1, 'word': 'El', 'start': 0, 'end': 2}, {'entity': 'I-PER', 'score': np.float32(0.99876237), 'index': 2, 'word': '##on', 'start': 2, 'end': 4}, {'entity': 'I-PER', 'score': np.float32(0.99889475), 'index': 3, 'word': 'Mu', 'start': 5, 'end': 7}, {'entity': 'I-PER', 'score': np.float32(0.9955519), 'index': 4, 'word': '##sk', 'start': 7, 'end': 9}, {'entity': 'I-ORG', 'score': np.float32(0.99919814), 'index': 6, 'word': 'Space', 'start': 18, 'end': 23}, {'entity': 'I-ORG', 'score': np.float32(0.99900395), 'index': 7, 'word': '##X', 'start': 23, 'end': 24}, {'entity': 'I-ORG', 'score': np.float32(0.9995678), 'index': 11, 'word': 'Te', 'start': 37, 'end': 39}, {'entity': 'I-ORG', 'score': np.float32(0.999033), 'index': 12, 'word': '##sla', 'start': 39, 'end': 42}, {'entity': 'I-ORG', 'score': np.float32(0.999501), 'index': 13, 'word': 'Motors', 'start': 43, 'end': 49}]
    


```python
from transformers import pipeline



ner=pipeline(task='ner',grouped_entities=True)

text='Elon Musk founded SpaceX in 2002 and Tesla Motors in 2003'
entities=ner(text)
print(entities)

```

    No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english and revision 4c53496 (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    config.json:   0%|          | 0.00/998 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/1.33G [00:00<?, ?B/s]


    Some weights of the model checkpoint at dbmdz/bert-large-cased-finetuned-conll03-english were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
    - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    


    tokenizer_config.json:   0%|          | 0.00/60.0 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]


    Device set to use cuda:0
    /usr/local/lib/python3.11/dist-packages/transformers/pipelines/token_classification.py:186: UserWarning: `grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to `aggregation_strategy="AggregationStrategy.SIMPLE"` instead.
      warnings.warn(
    

    [{'entity_group': 'PER', 'score': np.float32(0.9981639), 'word': 'Elon Musk', 'start': 0, 'end': 9}, {'entity_group': 'ORG', 'score': np.float32(0.99910104), 'word': 'SpaceX', 'start': 18, 'end': 24}, {'entity_group': 'ORG', 'score': np.float32(0.99936724), 'word': 'Tesla Motors', 'start': 37, 'end': 49}]
    


```python
from transformers import pipeline



ner=pipeline(task='ner',grouped_entities=True)

text='Elon Musk founded SpaceX in 2002 and Tesla Motors in 2003'
entities=ner(text)
print(entities)

```

    No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english and revision 4c53496 (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Some weights of the model checkpoint at dbmdz/bert-large-cased-finetuned-conll03-english were not used when initializing BertForTokenClassification: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight']
    - This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Device set to use cuda:0
    

    [{'entity_group': 'PER', 'score': np.float32(0.9981639), 'word': 'Elon Musk', 'start': 0, 'end': 9}, {'entity_group': 'ORG', 'score': np.float32(0.99910104), 'word': 'SpaceX', 'start': 18, 'end': 24}, {'entity_group': 'ORG', 'score': np.float32(0.99936724), 'word': 'Tesla Motors', 'start': 37, 'end': 49}]
    


```python

```

#Mask


```python
from transformers import pipeline

fill_mask=pipeline(task="fill-mask",model="bert-base-uncased")
text="[MASK] is The capital of India "
predictions=fill_mask(text)
print(predictions)
```

    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Device set to use cuda:0
    

    [{'score': 0.18057487905025482, 'token': 6768, 'token_str': 'delhi', 'sequence': 'delhi is the capital of india'}, {'score': 0.1614084541797638, 'token': 2009, 'token_str': 'it', 'sequence': 'it is the capital of india'}, {'score': 0.08124436438083649, 'token': 8955, 'token_str': 'mumbai', 'sequence': 'mumbai is the capital of india'}, {'score': 0.08072150498628616, 'token': 13624, 'token_str': 'hyderabad', 'sequence': 'hyderabad is the capital of india'}, {'score': 0.06982014328241348, 'token': 23571, 'token_str': 'lucknow', 'sequence': 'lucknow is the capital of india'}]
    


```python
from transformers import pipeline

fill_mask=pipeline(task="fill-mask",model="bert-base-uncased")
text="Capital of America is [MASK] "
predictions=fill_mask(text)
print(predictions)
```

    Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
    - This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Device set to use cuda:0
    

    [{'score': 0.7236712574958801, 'token': 1012, 'token_str': '.', 'sequence': 'capital of america is.'}, {'score': 0.14907433092594147, 'token': 1025, 'token_str': ';', 'sequence': 'capital of america is ;'}, {'score': 0.06741386651992798, 'token': 1064, 'token_str': '|', 'sequence': 'capital of america is |'}, {'score': 0.030373698100447655, 'token': 999, 'token_str': '!', 'sequence': 'capital of america is!'}, {'score': 0.028260309249162674, 'token': 1029, 'token_str': '?', 'sequence': 'capital of america is?'}]
    


```python
from transformers import pipeline

fill_mask=pipeline(task="fill-mask")
text="Capital of America is <mask>"
predictions=fill_mask(text)
print(predictions)
```

    No model was supplied, defaulted to distilbert/distilroberta-base and revision fb53ab8 (https://huggingface.co/distilbert/distilroberta-base).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Some weights of the model checkpoint at distilbert/distilroberta-base were not used when initializing RobertaForMaskedLM: ['roberta.pooler.dense.bias', 'roberta.pooler.dense.weight']
    - This IS expected if you are initializing RobertaForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Device set to use cuda:0
    

    [{'score': 0.0764598548412323, 'token': 20260, 'token_str': ' toast', 'sequence': 'Capital of America is toast'}, {'score': 0.026086673140525818, 'token': 1367, 'token_str': ' closed', 'sequence': 'Capital of America is closed'}, {'score': 0.025735294446349144, 'token': 16921, 'token_str': ' shrinking', 'sequence': 'Capital of America is shrinking'}, {'score': 0.023722056299448013, 'token': 3172, 'token_str': ' closing', 'sequence': 'Capital of America is closing'}, {'score': 0.017050106078386307, 'token': 27588, 'token_str': ' crumbling', 'sequence': 'Capital of America is crumbling'}]
    

#Text Generation Code


```python
from transformers import pipeline

generator=pipeline(task="text-generation")

prompt="once upon a time there was fox"

generated_text=generator(prompt,max_length=50,num_return_sequences=2)
print(generated_text)
```

    No model was supplied, defaulted to openai-community/gpt2 and revision 607a30d (https://huggingface.co/openai-community/gpt2).
    Using a pipeline without specifying a model name and revision in production is not recommended.
    Device set to use cuda:0
    Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
    Both `max_new_tokens` (=256) and `max_length`(=50) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    

    [{'generated_text': 'once upon a time there was foxes, leopards, and the like, that were to be found in the forests, far away from the great cities of India. The forest was a place of no more than a thousand acres, and was a place of immense beauty and of deep interest. The people, who had all their life toil in the forest, had scarcely seen any fox. Yet in the middle of the night, if one was at leisure, a man came to a place where there was a huge, square forest, and he asked if there was any fox there. "Well," said the man, "as much as you will admit, it is a fox." "It is a fox!" exclaimed the man, "that is to say, a dog, that is to say, a dog-head, and that is to say, a fox-beast." "I will not be able to tell you that a dog is a dog," said the man, "and that you would love to know it. And you will find, by the way, that it is a fox-scout." "That is not a dog-head, either," said the man, "as a fox-hound-dog is a fox-hound-breeder." "But you'}, {'generated_text': "once upon a time there was foxing. There was a young lady who was a long-time member of the royal family and she was also a woman. She looked pretty, she was very kind and it was like she had been looking for something. I saw her and I was like, 'What did she say to me?' I said I did! That's when she said, 'I do not know what she said to me so I will not talk to you until I hear it.' And so she came out and she looked at me and she said, 'What did you say?' So I said, 'I told you she was a girl. I will tell you what I told you.' She said, 'Well, I have been looking for her since when she was a girl. She has been searching for me for a long time.' So I was like, 'You know what I really want to know, because I have been looking for her for a long time.' So I was like, 'Well that's it.' She said, 'I have to say I never expected you to be so kind to me.' So I said, 'Yes, it is. I really do.' And she said, 'So do I.' So I said, 'Well, I have been looking"}]
    

#Feature Extraction


```python
from transformers import pipeline



feature_extractor=pipeline(task="feature-extraction",model="bert-base-uncased")

text="Mehfuz is very curious guy"
feature=feature_extractor(text,return_tensors="pt") #pt->pytorch | tensor

print(feature.shape)
print(feature)
```

    Device set to use cuda:0
    

    torch.Size([1, 10, 768])
    tensor([[[-0.0644,  0.3072, -0.2403,  ..., -0.2252,  0.4909,  0.3453],
             [ 0.0946, -0.0482, -0.5109,  ..., -0.8580,  0.9796,  0.2480],
             [ 0.1433,  0.4354,  0.0365,  ..., -1.1490, -0.4520, -0.8382],
             ...,
             [ 0.4254,  0.0395,  0.3553,  ..., -0.0688,  0.3166, -0.3482],
             [-0.2445, -0.5005,  0.0497,  ...,  0.5193,  0.5687, -0.3280],
             [ 0.7259,  0.2183, -0.0821,  ...,  0.2618, -0.5756, -0.3668]]])
    


```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = [
    "Mehfuz is very curious guy.",
    "He loves learning new things every day."
]

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)

# Get last hidden states (features)
features = outputs.last_hidden_state
print(features.shape)  # (batch_size, seq_len, hidden_size)
print(features)

```

    torch.Size([2, 11, 768])
    tensor([[[-0.2152,  0.1030, -0.2293,  ..., -0.3293,  0.6107,  0.4966],
             [ 0.0065,  0.2570, -0.4677,  ..., -1.0981,  0.8092,  0.1457],
             [ 0.0104,  0.5732,  0.2129,  ..., -1.1941, -0.6981, -0.7373],
             ...,
             [-0.0462, -0.0240,  0.7333,  ..., -0.0728,  0.1586, -0.6160],
             [-0.0725, -0.1594,  0.3120,  ...,  0.4188,  0.0967, -0.6107],
             [ 0.3023,  0.0615,  0.4857,  ...,  0.5289, -0.1642, -0.3487]],
    
            [[ 0.1842,  0.4473, -0.0721,  ..., -0.2856,  0.5538,  0.2169],
             [ 0.5064,  0.0792, -0.0768,  ..., -0.0598,  1.2614, -0.3738],
             [ 0.8088,  0.6948,  0.7463,  ..., -0.0521,  0.5078, -0.0668],
             ...,
             [ 0.7548,  0.2545, -0.1094,  ...,  0.3407, -0.3656, -0.5227],
             [ 0.9808,  0.2626,  0.2601,  ...,  0.3280, -0.5331, -0.4088],
             [ 0.6813,  0.2631,  0.1256,  ...,  0.2393,  0.1491, -0.0665]]],
           grad_fn=<NativeLayerNormBackward0>)
    

#Zero Shot Classification


```python
from transformers import pipeline


classifier=pipeline("zero-shot-classification",model="facebook/bart-large-mnli")

text="I have leno laptop which is very old now even the performance is not so good"
candidate_labels=["tech","sports","politics"]


result=classifier(text,candidate_labels=["tech","sports","politics"])

print(result)
```

    /usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
    The secret `HF_TOKEN` does not exist in your Colab secrets.
    To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
    You will be able to reuse this secret in all of your notebooks.
    Please note that authentication is recommended but still optional to access public models or datasets.
      warnings.warn(
    


    config.json: 0.00B [00:00, ?B/s]



    model.safetensors:   0%|          | 0.00/1.63G [00:00<?, ?B/s]



    tokenizer_config.json:   0%|          | 0.00/26.0 [00:00<?, ?B/s]



    vocab.json: 0.00B [00:00, ?B/s]



    merges.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]


    Device set to use cuda:0
    

    {'sequence': 'I have leno laptop which is very old now even the performance is not so good', 'labels': ['tech', 'sports', 'politics'], 'scores': [0.9480116367340088, 0.027083642780780792, 0.024904659017920494]}
    


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
