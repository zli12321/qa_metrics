import requests
import os
from .utils.tools import download_link

def download(model_name):
    # if model_name.lower() == 'bert':
    #     # Download BERT model
    #     model_url = 'https://drive.google.com/file/d/1ena_zUd42ju_5R3wKBidKdKuJYmF-IE_/view?usp=sharing'
    #     tokenizer_url = 'https://drive.google.com/file/d/1gpmCgHKX-zLpShDr-kDrotS0Fk6F8Ht1/view?usp=sharing'
    #     tokenizer_config_url = 'https://drive.google.com/file/d/1kxMwo9dD-gzgDuZMZAP64GmD_KLTofUh/view?usp=sharing'
    #     config_url = 'https://drive.google.com/file/d/14d3X6BbrDrKYlnwOYBDlW7_sjK5u2Msw/view?usp=sharing'
    #     vocab_url = 'https://drive.google.com/file/d/1N_kiKtJFvsIv8cwb-6ZOwXpqOi6ZLEEl/view?usp=sharing'

    #     model_dir = os.path.join(os.path.dirname(__file__), 'transformer_models/bert')
    #     model_path = os.path.join(model_dir, 'ae_tuned_bert.bin')
    #     tokenizer_path = os.path.join(model_dir, 'tokenizer.json')
    #     tokenizer_config_path = os.path.join(model_dir, 'tokenizer_config.json')
    #     config_path = os.path.join(model_dir, 'config.json')
    #     vocab_path = os.path.join(model_dir, 'vocab.txt')

    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)

    #     download_link(model_path, model_url, 'Fine-tuned BERT model')
    #     download_link(tokenizer_path, tokenizer_url, 'BERT tokenizer')
    #     download_link(tokenizer_config_path, tokenizer_config_url, 'BERT tokenizer config')
    #     download_link(config_path, config_url, 'BERT config')
    #     download_link(vocab_path, vocab_url, 'BERT vocab')        
        
        
    if model_name.lower() == 'cfm':
        clf_url = 'https://github.com/zli12321/qa_metrics/raw/master/qa_metrics/classifier/lr_classifier'
        vectorizer_url = 'https://github.com/zli12321/qa_metrics/raw/master/qa_metrics/classifier/tf-idf_vectorizer'
        model_dir = os.path.join(os.path.dirname(__file__), 'classifier')
        model_path = os.path.join(model_dir, 'lr_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'tf-idf_vectorizer.pkl')

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        download_link(model_path, clf_url, 'CF Matching model')
        download_link(vectorizer_path, vectorizer_url, 'CF Matching model tokenizer')
