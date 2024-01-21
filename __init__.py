# qa_evaluators/__init__.py

import requests
import os
from .qa_metrics.em import em_match
from .qa_metrics.f1 import f1_match
from .qa_metrics.cfm import CFMatcher
from .qa_metrics.transformerMatcher import TransformerMatcher  # assuming bem.py contains a function to download BERT

def download(model_name):
    if model_name.lower() == 'bert':
        # Download BERT model
        url = 'https://drive.google.com/file/d/1ena_zUd42ju_5R3wKBidKdKuJYmF-IE_/view?usp=sharing'
        model_dir = os.path.join(os.path.dirname(__file__), 'transformer_models')
        model_path = os.path.join(model_dir, 'ae_tuned_bert.bin')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not os.path.isfile(model_path):
            print("Downloading BERT Matching model...")
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                print("Download complete.")
            else:
                print("Failed to download the model. Status code:", response.status_code)
        # else:
        #     print("BERT model already exists.")

        return model_path
    if model_name.lower() == 'cfm':
        clf_url = 'https://github.com/zli12321/qa_metrics/raw/master/qa_metrics/classifier/lr_classifier'
        vectorizer_url = 'https://github.com/zli12321/qa_metrics/raw/master/qa_metrics/classifier/tf-idf_vectorizer'
        model_dir = os.path.join(os.path.dirname(__file__), 'classifier')
        model_path = os.path.join(model_dir, 'lr_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'tf-idf_vectorizer.pkl')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        if not os.path.isfile(model_path):
            print("Downloading CF Matching model...")
            response = requests.get(clf_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    f.write(response.content)
                print("Download clf model complete.")
            else:
                print("Failed to download the model. Status code:", response.status_code)

        if not os.path.isfile(vectorizer_path):
            print("Downloading CF Matching model tokenizer...")
            response = requests.get(vectorizer_url, stream=True)
            if response.status_code == 200:
                with open(vectorizer_path, 'wb') as f:
                    f.write(response.content)
                print("Download clf model complete.")
            else:
                print("Failed to download the model. Status code:", response.status_code)