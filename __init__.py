# qa_evaluators/__init__.py

import requests
import os
from .qa_metrics.em import em_match
from .qa_metrics.f1 import f1_match
from .qa_metrics.cfm import CFMatcher
from .qa_metrics.bem import bem  # assuming bem.py contains a function to download BERT

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
        else:
            print("BERT model already exists.")

        return model_path
