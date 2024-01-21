from .f1 import f1_score_with_precision_recall
from .utils.tools import normalize_answer, download_link
import joblib
from scipy.sparse import hstack
import numpy as np
import os


class CFMatcher:
    def __init__(self):
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.join(os.path.dirname(__file__), 'classifier')
        model_path = os.path.join(model_dir, 'lr_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'tf-idf_vectorizer.pkl')
        if not os.path.exists(model_path):
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            clf_url = 'https://github.com/zli12321/qa_metrics/raw/master/qa_metrics/classifier/lr_classifier'
            vectorizer_url = 'https://github.com/zli12321/qa_metrics/raw/master/qa_metrics/classifier/tf-idf_vectorizer'
            download_link(model_path, clf_url, 'CF Matching model')
            download_link(vectorizer_path, vectorizer_url, 'CF Matching model tokenizer')


        self.model = joblib.load(os.path.join(current_dir, 'classifier', 'lr_classifier.pkl'))
        self.tokenizer = joblib.load(os.path.join(current_dir, 'classifier', 'tf-idf_vectorizer.pkl'))

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
    are lists. 
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
    Input your reference and candidate answers, and the question.
    Return True if the candidate answer is deemed correct. Else, False.
    '''
    def cf_match(self, reference, candidate, question):
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