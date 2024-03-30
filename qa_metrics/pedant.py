from .f1 import f1_score_with_precision_recall
from .utils.tools import normalize_answer, download_link
import joblib
from scipy.sparse import hstack
import numpy as np
import os
import requests


class PEDANT:
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.join(current_dir, 'classifier')
        model_path = os.path.join(model_dir, 'lr_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'tf-idf_vectorizer.pkl')

        # Corrected URLs to point directly to the raw content
        clf_url = 'https://raw.githubusercontent.com/zli12321/qa_metrics/master/qa_metrics/classifier/lr_classifier.pkl'
        vectorizer_url = 'https://raw.githubusercontent.com/zli12321/qa_metrics/master/qa_metrics/classifier/tf-idf_vectorizer.pkl'

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        try:
            clf_updated = vectorizer_updated = False

            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                # Fetch headers to check last modified times without downloading the whole file
                clf_response = requests.head(clf_url)
                vectorizer_response = requests.head(vectorizer_url)

                clf_last_modified = clf_response.headers.get('Last-Modified')
                vectorizer_last_modified = vectorizer_response.headers.get('Last-Modified')

                if clf_last_modified and vectorizer_last_modified:
                    clf_local_modified = os.path.getmtime(model_path)
                    vectorizer_local_modified = os.path.getmtime(vectorizer_path)

                    clf_last_modified_dt = requests.utils.parsedate_to_datetime(clf_last_modified)
                    vectorizer_last_modified_dt = requests.utils.parsedate_to_datetime(vectorizer_last_modified)

                    # Compare remote file's last modified time with local file's last modified time
                    if clf_last_modified_dt.timestamp() > clf_local_modified:
                        clf_updated = True

                    if vectorizer_last_modified_dt.timestamp() > vectorizer_local_modified:
                        vectorizer_updated = True

            # Download updated models if necessary
            if clf_updated or not os.path.exists(model_path):
                print('Downloading updated PANDA model...')
                download_link(model_path, clf_url, 'PANDA model')

            if vectorizer_updated or not os.path.exists(vectorizer_path):
                print('Downloading updated PANDA evaluation model tokenizer...')
                download_link(vectorizer_path, vectorizer_url, 'PANDA evaluation model tokenizer')

        except requests.ConnectionError:
            print("No internet connection. Using existing models.")
        except requests.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
        except Exception as err:
            print(f"An unexpected error occurred: {err}")

        self.model = joblib.load(model_path)
        self.tokenizer = joblib.load(vectorizer_path)


    def download_latest_model(self):
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.join(current_dir, 'classifier')
        model_path = os.path.join(model_dir, 'lr_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'tf-idf_vectorizer.pkl')

        # Corrected URLs to point directly to the raw content
        clf_url = 'https://raw.githubusercontent.com/zli12321/qa_metrics/master/qa_metrics/classifier/lr_classifier.pkl'
        vectorizer_url = 'https://raw.githubusercontent.com/zli12321/qa_metrics/master/qa_metrics/classifier/tf-idf_vectorizer.pkl'

        print('Downloading updated PANDA model...')
        download_link(model_path, clf_url, 'PANDA model')

        print('Downloading updated PANDA evaluation model tokenizer...')
        download_link(vectorizer_path, vectorizer_url, 'PANDA evaluation model tokenizer')

        self.model = joblib.load(model_path)
        self.tokenizer = joblib.load(vectorizer_path)

    '''
    Return the confidence score between the reference and candidate answers. 
    reference, candidate, and question are strings.
    '''
    def get_score(self, reference, candidate, question):
        reference = normalize_answer(str(reference))
        candidate = normalize_answer(str(candidate))
        question = normalize_answer(str(question))

        input_texts = []
        f1_scores, precisions, recalls = [], [], []
        input_texts.append("[CLS] " + candidate + " [SEP] " + reference + " [SEP] " + question + " [SEP]")
        f1_results = f1_score_with_precision_recall(reference, candidate)
        f, p, r = f1_results['f1'], f1_results['precision'], f1_results['recall']
        f1_scores.append(f)
        precisions.append(p)
        recalls.append(r)

        f1_scores = np.array(f1_scores).reshape(-1, 1)
        precisions = np.array(precisions).reshape(-1, 1)
        recalls = np.array(recalls).reshape(-1, 1)

        texts = self.tokenizer.transform(input_texts)

        '''
        Concatenate text features with f1 features
        '''
        features = hstack([texts, f1_scores, precisions, recalls])
        pred_probas = self.model.predict_proba(features)

        return pred_probas[0][0]


    '''
    Returns the classifier confidence score for the candidate answer matching judgment if the reference and candidate answers
    are lists. The reference and candidate answers can lists of strings or just strings. The question is a string.
    '''
    def get_scores(self, reference, candidate, question):
        # Calculate the F1 score between the referee and candidate
        confidece_scores = {}
        if isinstance(reference, list) and isinstance(candidate, list):
            references = [normalize_answer(str(ele)) for ele in reference]
            candidates = [normalize_answer(str(ele)) for ele in candidate]
            question = normalize_answer(str(question))

            for candidate in candidates:
                input_texts = []
                f1_scores, precisions, recalls = [], [], []
                for reference in references:
                    input_texts.append("[CLS] " + candidate + " [SEP] " + reference + " [SEP] " + question + " [SEP]")
                    f1_results = f1_score_with_precision_recall(reference, candidate)
                    f, p, r = f1_results['f1'], f1_results['precision'], f1_results['recall']
                    f1_scores.append(f)
                    precisions.append(p)
                    recalls.append(r)

                f1_scores = np.array(f1_scores).reshape(-1, 1)
                precisions = np.array(precisions).reshape(-1, 1)
                recalls = np.array(recalls).reshape(-1, 1)

                texts = self.tokenizer.transform(input_texts)

                '''
                Concatenate text features with f1 features
                '''
                features = hstack([texts, f1_scores, precisions, recalls])
                pred_probas = self.model.predict_proba(features)

                for i in range(len(pred_probas)):
                    if references[i] not in confidece_scores:
                        confidece_scores[references[i]] = {}
                    confidece_scores[references[i]][candidate] = pred_probas[i][0]
                            
            return confidece_scores
        elif isinstance(reference, list):
            references = [normalize_answer(str(ele)) for ele in reference]
            candidates = normalize_answer(str(candidate))
            question = normalize_answer(str(question))

            input_texts = []
            f1_scores, precisions, recalls = [], [], []
            for reference in references:
                input_texts.append("[CLS] " + candidate + " [SEP] " + reference + " [SEP] " + question + " [SEP]")
                f1_results = f1_score_with_precision_recall(reference, candidate)
                f, p, r = f1_results['f1'], f1_results['precision'], f1_results['recall']
                f1_scores.append(f)
                precisions.append(p)
                recalls.append(r)

            f1_scores = np.array(f1_scores).reshape(-1, 1)
            precisions = np.array(precisions).reshape(-1, 1)
            recalls = np.array(recalls).reshape(-1, 1)

            texts = self.tokenizer.transform(input_texts)

            '''
            Concatenate text features with f1 features
            '''
            features = hstack([texts, f1_scores, precisions, recalls])
            pred_probas = self.model.predict_proba(features)

            
            for i in range(len(pred_probas)):
                confidece_scores[references[i]] = {}
                confidece_scores[references[i]][candidate] = pred_probas[i][0]
                            
            return confidece_scores
        elif isinstance(candidate, list):
            candidates = [normalize_answer(str(ele)) for ele in candidate]
            reference = normalize_answer(str(reference))
            question = normalize_answer(str(question))

            input_texts = []
            f1_scores, precisions, recalls = [], [], []
            for candidate in candidates:
                input_texts.append("[CLS] " + candidate + " [SEP] " + reference + " [SEP] " + question + " [SEP]")
                f1_results = f1_score_with_precision_recall(reference, candidate)
                f, p, r = f1_results['f1'], f1_results['precision'], f1_results['recall']
                f1_scores.append(f)
                precisions.append(p)
                recalls.append(r)

            f1_scores = np.array(f1_scores).reshape(-1, 1)
            precisions = np.array(precisions).reshape(-1, 1)
            recalls = np.array(recalls).reshape(-1, 1)

            texts = self.tokenizer.transform(input_texts)

            '''
            Concatenate text features with f1 features
            '''
            features = hstack([texts, f1_scores, precisions, recalls])
            pred_probas = self.model.predict_proba(features)
            
            confidece_scores[reference] = {}
            for i in range(len(pred_probas)):
                confidece_scores[reference][candidates[i]] = pred_probas[i][0]
                            
            return confidece_scores
        else:
            confidece_scores[reference] = {}
            confidece_scores[reference][candidate] = self.get_score(reference, candidate, question)

            return confidece_scores

    '''
    Given a list of reference, candidate, and a question, return the pair with the highest confidence score.
    '''
    def get_highest_score(self, reference, candidate, question):
        confidence_scores = self.get_scores(reference, candidate, question)

        max_score = -1
        max_pair = (None, None)

        for reference, candidates in confidence_scores.items():
            for candidate, score in candidates.items():
                if score > max_score:
                    max_score = score
                    max_pair = (reference, candidate)

        return max_pair, max_score

    '''
    Input your reference and candidate answers, and the question.
    Return True if the candidate answer is deemed correct. Else, False.
    '''
    def evaluate(self, reference, candidate, question):
        # Calculate the F1 score between the referee and candidate
        judgment = False
        if isinstance(reference, list) and isinstance(candidate, list):
            references = [normalize_answer(str(ele)) for ele in reference]
            candidates = [normalize_answer(str(ele)) for ele in candidate]
            question = normalize_answer(str(question))

            for candidate in candidates:
                if judgment == False:
                    input_texts = []
                    f1_scores, precisions, recalls = [], [], []
                    for reference in references:
                        input_texts.append("[CLS] " + candidate + " [SEP] " + reference + " [SEP] " + question + " [SEP]")
                        f1_results = f1_score_with_precision_recall(reference, candidate)
                        f, p, r = f1_results['f1'], f1_results['precision'], f1_results['recall']
                        f1_scores.append(f)
                        precisions.append(p)
                        recalls.append(r)

                        f1_scores = np.array(f1_scores).reshape(-1, 1)
                        precisions = np.array(precisions).reshape(-1, 1)
                        recalls = np.array(recalls).reshape(-1, 1)

                        texts = self.tokenizer.transform(input_texts)

                    '''
                    Concatenate text features with f1 features
                    '''
                    features = hstack([texts, f1_scores, precisions, recalls])
                    preds = self.model.predict(features)

                    if "correct" in preds:
                        judgment = True
                            
            return judgment
        elif isinstance(reference, list):
            references = [normalize_answer(str(ele)) for ele in reference]
            candidates = normalize_answer(str(candidate))
            question = normalize_answer(str(question))

            input_texts = []
            f1_scores, precisions, recalls = [], [], []
            for reference in references:
                input_texts.append("[CLS] " + candidate + " [SEP] " + reference + " [SEP] " + question + " [SEP]")
                f1_results = f1_score_with_precision_recall(reference, candidate)
                f, p, r = f1_results['f1'], f1_results['precision'], f1_results['recall']
                f1_scores.append(f)
                precisions.append(p)
                recalls.append(r)

            f1_scores = np.array(f1_scores).reshape(-1, 1)
            precisions = np.array(precisions).reshape(-1, 1)
            recalls = np.array(recalls).reshape(-1, 1)

            texts = self.tokenizer.transform(input_texts)

            '''
            Concatenate text features with f1 features
            '''
            features = hstack([texts, f1_scores, precisions, recalls])
            preds = self.model.predict(features)

            if "correct" in preds:
                judgment = True
                            
            return judgment
        elif isinstance(candidate, list):
            candidates = [normalize_answer(str(ele)) for ele in candidate]
            reference = normalize_answer(str(reference))
            question = normalize_answer(str(question))

            input_texts = []
            f1_scores, precisions, recalls = [], [], []
            for candidate in candidates:
                input_texts.append("[CLS] " + candidate + " [SEP] " + reference + " [SEP] " + question + " [SEP]")
                f1_results = f1_score_with_precision_recall(reference, candidate)
                f, p, r = f1_results['f1'], f1_results['precision'], f1_results['recall']
                f1_scores.append(f)
                precisions.append(p)
                recalls.append(r)

            f1_scores = np.array(f1_scores).reshape(-1, 1)
            precisions = np.array(precisions).reshape(-1, 1)
            recalls = np.array(recalls).reshape(-1, 1)

            texts = self.tokenizer.transform(input_texts)

            '''
            Concatenate text features with f1 features
            '''
            features = hstack([texts, f1_scores, precisions, recalls])
            preds = self.model.predict(features)

            if "correct" in preds:
                judgment = True
                            
            return judgment
        else:
            reference = normalize_answer(str(reference))
            candidate = normalize_answer(str(candidate))
            question = normalize_answer(str(question))

            input_texts = []
            f1_scores, precisions, recalls = [], [], []
            input_texts.append("[CLS] " + candidate + " [SEP] " + reference + " [SEP] " + question + " [SEP]")
            f1_results = f1_score_with_precision_recall(reference, candidate)
            f, p, r = f1_results['f1'], f1_results['precision'], f1_results['recall']
            f1_scores.append(f)
            precisions.append(p)
            recalls.append(r)

            f1_scores = np.array(f1_scores).reshape(-1, 1)
            precisions = np.array(precisions).reshape(-1, 1)
            recalls = np.array(recalls).reshape(-1, 1)

            texts = self.tokenizer.transform(input_texts)

            '''
            Concatenate text features with f1 features
            '''
            features = hstack([texts, f1_scores, precisions, recalls])
            preds = self.model.predict(features)

            if "correct" in preds:
                judgment = True
                            
            return judgment