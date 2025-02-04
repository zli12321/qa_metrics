from .utils.tools import *
import joblib
from scipy.sparse import hstack
import numpy as np
import os


class PEDANT:
    def __init__(self, fast_eval=True, fast_model=False): 
        current_dir = os.path.dirname(__file__)
        model_dir = os.path.join(current_dir, 'classifier') 
        model_path = os.path.join(model_dir, 'lr_classifier.pkl') 
        vectorizer_path = os.path.join(model_dir, 'tf-idf_vectorizer.pkl') 
        rule_clf_path = os.path.join(model_dir, 'rule_classifier.pkl') 
        type_clf_path = os.path.join(model_dir, 'type_classifier.pkl') 
        light_model_path = os.path.join(model_dir, 'light_lr_classifier.pkl') 
        light_vectorizer_path = os.path.join(model_dir, 'light_tf-idf_vectorizer.pkl') 


        clf_url = 'https://github.com/zli12321/pedant_models/raw/refs/heads/main/lr_classifier' 
        rule_clf_url = 'https://github.com/zli12321/pedant_models/raw/refs/heads/main/rule_classifier'
        type_clf_url = 'https://github.com/zli12321/pedant_models/raw/refs/heads/main/type_classifier'
        vectorizer_url = 'https://github.com/zli12321/pedant_models/raw/refs/heads/main/tf-idf_vectorizer'
        light_clf_url = ''
        light_vectorizer_url = ''

        self.fast_eval = fast_eval
        self.fast_model = fast_model

        if True:
            if not os.path.exists(model_dir): 
                os.makedirs(model_dir)
            try:
                # print('Downloaded model path: ', model_path) 
                download_link(model_path, clf_url, 'PEDANT model') 
                download_link(vectorizer_path, vectorizer_url, 'Tokenizer') 
                download_link(rule_clf_path, rule_clf_url, 'Rule feature extractor') 
                download_link(type_clf_path, type_clf_url, 'Type feature extractor') 
                download_link(light_model_path, light_clf_url, 'Light PEDANT model') 
                download_link(light_vectorizer_path, light_vectorizer_url, 'Light Tokenizer') 
                
            except:
                pass


        self.model = joblib.load(model_path)
        self.tokenizer = joblib.load(vectorizer_path)
        self.rule_model = joblib.load(rule_clf_path)
        self.type_model = joblib.load(type_clf_path)

        if self.fast_model == True:
            self.light_model = joblib.load(light_model_path)
            self.light_tokenizer = joblib.load(light_vectorizer_path)

    def download_latest_model(self):
        current_dir = os.path.dirname(__file__) 
        model_dir = os.path.join(current_dir, 'classifier') 
        model_path = os.path.join(model_dir, 'lr_classifier.pkl') 
        vectorizer_path = os.path.join(model_dir, 'tf-idf_vectorizer.pkl') 
        rule_clf_path = os.path.join(model_dir, 'rule_classifier.pkl') 
        type_clf_path = os.path.join(model_dir, 'type_classifier.pkl') 

        if not os.path.exists(model_dir): 
            os.makedirs(model_dir) 

        clf_url = 'https://github.com/zli12321/pedant_models/raw/refs/heads/main/lr_classifier.pkl' 
        rule_clf_url = 'https://github.com/zli12321/pedant_models/raw/refs/heads/main/rule_classifier.pkl'
        type_clf_url = 'https://github.com/zli12321/pedant_models/raw/refs/heads/main/type_classifier.pkl'
        vectorizer_url = 'https://github.com/zli12321/pedant_models/raw/refs/heads/main/tf-idf_vectorizer.pkl' 

        print('Downloaded model path: ', model_path) 
        download_link(model_path, clf_url, 'PEDANT model') 
        download_link(vectorizer_path, vectorizer_url, 'Tokenizer') 
        download_link(rule_clf_path, rule_clf_url, 'Rule feature extractor') 
        download_link(type_clf_path, type_clf_url, 'Type feature extractor') 

        self.model = joblib.load(model_path)
        self.tokenizer = joblib.load(vectorizer_path)
        self.rule_model = joblib.load(rule_clf_path)
        self.type_model = joblib.load(type_clf_path)

    def get_rule_features(self, data):
        inputs = ['[CLS] ' + str(ele['question']) + ' [SEP] ' + str(ele['reference']) + ' [SEP] ' + str(ele['candidate']) + ' [SEP]' for ele in data]
        f1, p, r = self.get_f1_features(data)
        inputs = self.tokenizer.transform(inputs)
        return self.rule_model.predict_proba(hstack([inputs, f1, p, r])), self.rule_model.predict_log_proba(hstack([inputs, f1, p, r]))

    def get_type_features(self, data):
        inputs = ['[CLS] ' + str(ele['question']) + ' [SEP] ' + str(ele['reference']) + ' [SEP]' for ele in data]
        inputs = self.tokenizer.transform(inputs)
        return self.type_model.predict_proba(inputs), self.type_model.predict_log_proba(inputs)

    def get_f1_features(self, rule_data):
        f1_scores, precisions, recalls = [], [], []

        for i in range(len(rule_data)):
            f1, p, r = calculate_f1_score_with_precision(str(rule_data[i]['reference']), str(rule_data[i]['candidate']))
            f1_scores.append(f1)
            precisions.append(p)
            recalls.append(r)

        f1_scores=np.array(f1_scores).reshape(-1, 1)
        precisions=np.array(precisions).reshape(-1, 1)
        recalls=np.array(recalls).reshape(-1, 1)

        return f1_scores, precisions, recalls

    def contruct_light_features(self, data):
        in_texts = []
        for ele in data:
            # text = '[CLS] ' + lemmatize_text(normalize_answer(str(ele['question']))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ele['reference']))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ele['candidate']))) + ' [SEP]'
            
            text = '[CLS] ' + str(ele['question']) + ' [SEP] ' + str(ele['reference']) + ' [SEP] ' + str(ele['candidate']) + ' [SEP]'
            # input_text = vectorizer.transform(text)
            in_texts.append(text)

        # print(in_texts)
        in_texts = self.tokenizer.transform(in_texts)
        f1, p, r = self.get_f1_features(data)

       
        all_feats = hstack([f1, p, r, in_texts])


        return all_feats

    def construct_features(self, data, with_log_probas=False):
        in_texts = []
        for ele in data:
            # text = '[CLS] ' + lemmatize_text(normalize_answer(str(ele['question']))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ele['reference']))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ele['candidate']))) + ' [SEP]'
            
            text = '[CLS] ' + str(ele['question']) + ' [SEP] ' + str(ele['reference']) + ' [SEP] ' + str(ele['candidate']) + ' [SEP]'
            # input_text = vectorizer.transform(text)
            in_texts.append(text)

        # print(in_texts)
        in_texts = self.tokenizer.transform(in_texts)
        f1, p, r = self.get_f1_features(data)

        # print(f'f1 data: f: {f1}; p: {p}; r: {r}')
        rule_feats, rule_log = self.get_rule_features(data)
        type_feats, type_log = self.get_type_features(data)
        if with_log_probas==True:
            all_feats = hstack([rule_feats, rule_log, type_feats, type_log, in_texts, f1, p, r])
        else:
            all_feats = hstack([f1, p, r, rule_feats, type_feats, in_texts])
            # all_feats = hstack([type_feats, in_texts, f1, p, r])

        return all_feats

    '''
    Return the confidence score between the reference and candidate answers. 
    reference, candidate, and question are strings.
    '''
    def get_score(self, reference, candidate, question):
        if len(reference) == 0 or len(candidate) == 0:
            return 0

        if self.fast_eval == False:
            reference = lemmatize_text(normalize_answer(str(reference)))
            candidate = lemmatize_text(normalize_answer(str(candidate)))
            question = lemmatize_text(normalize_answer(str(question)))
        else:
            reference = normalize_answer(str(reference))
            candidate = normalize_answer(str(candidate))
            question = normalize_answer(str(question))

        if reference in candidate:
            return 1.0

        data = [{
            'reference': reference,
            'candidate': candidate,
            'question': question
        }]


        if self.fast_model == False:
            feature = self.construct_features(data)
            pred_probas = self.model.predict_proba(feature)
        else:
            feature = self.contruct_light_features(data)
            pred_probas = self.light_model.predict_proba(feature)
        
        return pred_probas[0][0]

    '''
    Returns the classifier confidence score for the candidate answer matching judgment if the reference and candidate answers
    are lists. The reference and candidate answers can lists of strings or just strings. The question is a string.
    '''
    def get_scores(self, reference, candidate, question):
        # Calculate the F1 score between the referee and candidate
        confidence_scores = {}
        if isinstance(reference, list) and isinstance(candidate, list):
            references = reference
            candidates = candidate

            
            for reference in references:
                for candidate in candidates:
                    score = self.get_score(reference, candidate, question)
                    confidence_scores[reference] = {}
                    confidence_scores[reference][candidate] = score
                            
            return confidence_scores

        elif isinstance(reference, list):
            references = reference
            for reference in references:
                score = self.get_score(reference, candidate, question)
                confidence_scores[reference] = {}
                confidence_scores[reference][candidate] = score
                            
            return confidence_scores
        elif isinstance(candidate, list):
            candidates = candidate
            for candidate in candidates:
                score = self.get_score(reference, candidate, question)
                confidence_scores[reference] = {}
                confidence_scores[reference][candidate] = score
                            
            return confidence_scores
        else:
            confidence_scores[reference] = {}
            confidence_scores[reference][candidate] = self.get_score(reference, candidate, question)

            return confidence_scores

    '''
    return the type of the question
    '''
    def get_question_type(self, reference, question):
        if len(reference) == 0:
            return ['Empty reference']
    
        if isinstance(reference, list):
            if self.fast_eval == False:
                inputs = ['[CLS] ' + lemmatize_text(normalize_answer(str(question))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ref))) + ' [SEP]' for ref in reference]
            else:
                inputs = ['[CLS] ' + normalize_answer(str(question)) + ' [SEP] ' + normalize_answer(str(ref)) + ' [SEP]' for ref in reference]
            inputs = self.tokenizer.transform(inputs)
            outs = self.type_model.predict(inputs)
            return list(set(outs))
        else:
            if self.fast_eval == False:
                inputs = ['[CLS] ' + lemmatize_text(normalize_answer(str(question))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(reference))) + ' [SEP]']
            else:
                inputs = ['[CLS] ' + normalize_answer(str(question)) + ' [SEP] ' + normalize_answer(str(reference)) + ' [SEP]']

            inputs = self.tokenizer.transform(inputs)
            return self.type_model.predict(inputs)[0]

    def get_judgement_rule(self, reference, candidate, question):
        if len(reference) == 0:
            return ['Empty reference']
        if len(candidate) == 0:
            return ['Empty candidate']
        
        if isinstance(reference, list):

            data = [{
                'question': question,
                'reference': ref,
                'candidate': candidate
            } for ref in reference]

            if self.fast_eval == False:
                inputs = ['[CLS] ' + lemmatize_text(normalize_answer(str(ele['question']))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ele['reference']))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ele['candidate']))) + ' [SEP]' for ele in data]
            else:
                inputs = ['[CLS] ' + normalize_answer(str(ele['question'])) + ' [SEP] ' + normalize_answer(str(ele['reference'])) + ' [SEP] ' + normalize_answer(str(ele['candidate'])) + ' [SEP]' for ele in data]
            
            f1, p, r = self.get_f1_features(data)
            inputs = self.tokenizer.transform(inputs)

            rules = [
                'Widely recognized aliases, pseudonyms that are commonly associated with referred answer entities are acceptable.',
                'Exact dates, years, numerical values are required unless the question specifically asks for approximations.',
                'The answer provides less detail but should include the essential and correct information required by the question (specificity level 1).',
                'The answer contains additional information that does not contradict the question or initial answer (specificity level 2).',
                'A high degree of word overlap does not establish equivalence. The answer must be contextually and semantically accurate.',
                'Any irrelevant or inaccurate description related to the question should be considered incorrect.',
                'The response is correct but not in the initially provided list.'
            ]

            out_rule_num = self.rule_model.predict(hstack([inputs, f1, p, r]))
            rules = [rules[idx] for idx in out_rule_num]
            return list(set(rules))
        else:

            data = [{
                'question': question,
                'reference': reference,
                'candidate': candidate
            }]

            if self.fast_eval == False:
                inputs = ['[CLS] ' + lemmatize_text(normalize_answer(str(ele['question']))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ele['reference']))) + ' [SEP] ' + lemmatize_text(normalize_answer(str(ele['candidate']))) + ' [SEP]' for ele in data]
            else:
                inputs = ['[CLS] ' + normalize_answer(str(ele['question'])) + ' [SEP] ' + normalize_answer(str(ele['reference'])) + ' [SEP] ' + normalize_answer(str(ele['candidate'])) + ' [SEP]' for ele in data]

            f1, p, r = self.get_f1_features(data)
            inputs = self.tokenizer.transform(inputs)

            rules = [
                'Widely recognized aliases, pseudonyms that are commonly associated with referred answer entities are acceptable.',
                'Exact dates, years, numerical values are required unless the question specifically asks for approximations.',
                'The answer provides less detail but should include the essential and correct information required by the question (specificity level 1).',
                'The answer contains additional information that does not contradict the question or initial answer (specificity level 2).',
                'A high degree of word overlap does not establish equivalence. The answer must be contextually and semantically accurate.',
                'Any irrelevant or inaccurate description related to the question should be considered incorrect.',
                'The response is correct but not in the initially provided list.'
            ]

            return rules[self.rule_model.predict(hstack([inputs, f1, p, r]))[0]-1]

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
        input_texts = []
        if isinstance(reference, list) and isinstance(candidate, list):
            if len(reference) == 0 or len(candidate) == 0:
                return False
            

            if self.fast_eval == False:
                references = [lemmatize_text(normalize_answer(str(ele))) for ele in reference]
                candidates = [lemmatize_text(normalize_answer(str(ele))) for ele in candidate]
                question = lemmatize_text(normalize_answer(str(question)))
            else:
                references = [normalize_answer(str(ele)) for ele in reference]
                candidates = [normalize_answer(str(ele)) for ele in candidate]
                question = normalize_answer(str(question))

            for candidate in candidates:
                if judgment == False:
                    for reference in references:
                        if reference in candidate:
                            return True

                        input_texts.append({'reference': reference,
                                            'candidate': candidate,
                                            'question': question
                                            })
        elif isinstance(reference, list):
            if len(reference) == 0 or len(candidate) == 0:
                return False
            
            if self.fast_eval == False:
                references = [lemmatize_text(normalize_answer(str(ele))) for ele in reference]
                candidate = lemmatize_text(normalize_answer(str(candidate)))
                question = lemmatize_text(normalize_answer(str(question)))
            else:
                references = [normalize_answer(str(ele)) for ele in reference]
                candidate = normalize_answer(str(candidate))
                question = normalize_answer(str(question))

            for reference in references:
                if reference in candidate:
                    return True
                
                input_texts.append({'reference': reference,
                                    'candidate': candidate,
                                    'question': question
                                     })
        elif isinstance(candidate, list):
            if len(reference) == 0 or len(candidate) == 0:
                return False
            

            if self.fast_eval == False:
                candidates = [lemmatize_text(normalize_answer(str(ele))) for ele in candidate]
                reference = lemmatize_text(normalize_answer(str(reference)))
                question = lemmatize_text(normalize_answer(str(question)))
            else:
                candidates = [normalize_answer(str(ele)) for ele in candidate]
                reference = normalize_answer(str(reference))
                question = normalize_answer(str(question))


            for candidate in candidates:
                if reference in candidate:
                    return True
                
                input_texts.append({'reference': reference,
                                    'candidate': candidate,
                                    'question': question
                                     })
        else:
            if len(reference) == 0 or len(candidate) == 0:
                return False
              
            
            if self.fast_eval == False:
                reference = lemmatize_text(normalize_answer(str(reference)))
                candidate = lemmatize_text(normalize_answer(str(candidate)))
                question = lemmatize_text(normalize_answer(str(question)))
            else:
                reference = normalize_answer(str(reference))
                candidate = normalize_answer(str(candidate))
                question = normalize_answer(str(question))

            if reference in candidate:
                return True

            input_texts.append({'reference': reference,
                                'candidate': candidate,
                                'question': question
                                })

        if self.fast_model == False:
            features = self.construct_features(input_texts)
            preds = self.model.predict(features)
        else:
            features = self.contruct_light_features(input_texts)
            preds = self.light_model.predict(features)




        if "correct" in preds:
            judgment = True


        return judgment