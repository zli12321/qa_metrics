import os
from .em import em_match
import torch

class TransformerMatcher:
    def __init__(self, model='zli12321/roberta-large-qa-evaluator'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        '''
        Fetch the model and tokenizer from the local directory
        '''
        current_dir = os.path.dirname(__file__)
        
        if model == 'bert':
            from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
            model_dir = os.path.join(current_dir, 'transformer_models/bert')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_path = 'zli12321/answer_equivalence_bert'
            config= BertConfig.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path, config=config, cache_dir=model_dir).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        if model == 'distilbert':
            from transformers import DistilBertForSequenceClassification, DistilBertConfig, DistilBertTokenizer
            model_dir = os.path.join(current_dir, 'transformer_models/distilbert')

            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_path = 'zli12321/answer_equivalence_distilbert'

            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            config = DistilBertConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path, config=config).to(self.device)
        elif model == 'distilroberta':
            from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
            model_dir = os.path.join(current_dir, 'transformer_models/distilroberta')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = 'zli12321/answer_equivalence_distilroberta'

            config = RobertaConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir=model_dir).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        elif model == 'roberta':
            from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
            model_dir = os.path.join(current_dir, 'transformer_models/roberta')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = 'zli12321/answer_equivalence_roberta'
            config = RobertaConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir=model_dir).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        elif model == 'roberta-large':
            from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
            model_dir = os.path.join(current_dir, 'transformer_models/roberta-large')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = 'zli12321/answer_equivalence_roberta-large'
            config = RobertaConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir=model_dir).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        elif model == 'tiny-bert':
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            model_dir = os.path.join(current_dir, 'transformer_models/tiny_bert')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_path = 'zli12321/answer_equivalence_tiny_bert'
            config= AutoModelForSequenceClassification.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, cache_dir=model_dir).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        else:
            from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
            model_dir = os.path.join(current_dir, 'transformer_models/roberta-large')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = 'zli12321/roberta-large-qa-evaluator'
            config = RobertaConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir=model_dir).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=model_dir)

            # model_path = model
            # if 'roberta' in model.lower():
            #     from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
            #     model_dir = os.path.join(current_dir, 'transformer_models/' + model_path)
                
            #     # Ensure the target directory exists
            #     if not os.path.exists(model_dir):
            #         os.makedirs(model_dir)

            #     config = RobertaConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            #     self.model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir=model_dir).to(self.device)
            #     self.tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=model_dir)
            # elif 'distilbert' in model.lower():
            #     from transformers import DistilBertForSequenceClassification, DistilBertConfig, DistilBertTokenizer
            #     model_dir = os.path.join(current_dir, 'transformer_models/distilbert')

            #     # Ensure the target directory exists
            #     if not os.path.exists(model_dir):
            #         os.makedirs(model_dir)
            #     self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            #     config = DistilBertConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            #     self.model = DistilBertForSequenceClassification.from_pretrained(model_path, config=config).to(self.device)
            # elif 'tiny-bert' in model.lower():
            #     from transformers import AutoTokenizer, AutoModelForSequenceClassification
            #     model_dir = os.path.join(current_dir, 'transformer_models/tiny_bert')
                
            #     # Ensure the target directory exists
            #     if not os.path.exists(model_dir):
            #         os.makedirs(model_dir)

            #     config= AutoModelForSequenceClassification.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            #     self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, cache_dir=model_dir).to(self.device)
            #     self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=model_dir)
            # elif 'bert' in model.lower():
            #     from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
            #     model_dir = os.path.join(current_dir, 'transformer_models/bert')
                
            #     # Ensure the target directory exists
            #     if not os.path.exists(model_dir):
            #         os.makedirs(model_dir)
                
            #     config= BertConfig.from_pretrained(model_path)
            #     self.model = BertForSequenceClassification.from_pretrained(model_path, config=config, cache_dir=model_dir).to(self.device)
            #     self.tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=model_dir)

    def download_latest_model(self, model='roberta'):
        current_dir = os.path.dirname(__file__)
        model_file = "model.safetensors"

        if model == 'bert':
            from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
            model_dir = os.path.join(current_dir, 'transformer_models/bert')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_path = 'zli12321/answer_equivalence_bert'
            config= BertConfig.from_pretrained(model_path)
            self.model = BertForSequenceClassification.from_pretrained(model_path, config=config, cache_dir=model_dir, model_file=model_file).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        if model == 'distilbert':
            from transformers import DistilBertForSequenceClassification, DistilBertConfig, DistilBertTokenizer
            model_dir = os.path.join(current_dir, 'transformer_models/distilbert')

            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            model_path = 'zli12321/answer_equivalence_distilbert'

            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            config = DistilBertConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = DistilBertForSequenceClassification.from_pretrained(model_path, config=config, model_file=model_file).to(self.device)
        elif model == 'distilroberta':
            from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
            model_dir = os.path.join(current_dir, 'transformer_models/distilroberta')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = 'zli12321/answer_equivalence_distilroberta'

            config = RobertaConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir=model_dir, model_file=model_file).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        elif model == 'roberta':
            from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
            model_dir = os.path.join(current_dir, 'transformer_models/roberta')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = 'zli12321/answer_equivalence_roberta'
            config = RobertaConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir=model_dir, model_file=model_file).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=model_dir)
        elif model == 'roberta-large':
            from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
            model_dir = os.path.join(current_dir, 'transformer_models/roberta-large')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            model_path = 'zli12321/answer_equivalence_roberta-large'
            config = RobertaConfig.from_pretrained(model_path, hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path, cache_dir=model_dir, model_file=model_file).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path, cache_dir=model_dir)

    '''
    Return the confidence score between the reference and candidate answers. 
    reference, candidate, and question are strings.
    '''
    def get_score(self, reference, candidate, question):
        if em_match(reference, candidate) == True:
            return 1.0
        input_text = "[CLS] " +str(candidate) + " [SEP] " +str(reference) + " [SEP] " +question + " [SEP]"
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
            )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # Apply sigmoid to the logits to get the probability score for each class
            probabilities = torch.sigmoid(outputs.logits).squeeze()

        # Convert to numpy
        probabilities = probabilities.cpu().numpy()

        # Assuming that you are interested in the second class (usually 'correct')
        # Make sure the index (here [1]) corresponds to the correct class in your case
        score = probabilities[1] if len(probabilities.shape) > 0 else probabilities

        return score
    
    '''
    Returns the classifier confidence score for the candidate answer matching judgment if the reference and candidate answers
    are lists. The reference and candidate answers can lists of strings or just strings. The question is a string.
    '''
    def get_scores(self, reference, candidate, question):
        # Calculate the F1 score between the referee and candidate
        confidece_scores = {}
        if isinstance(reference, list) and isinstance(candidate, list):
            references = [str(ele) for ele in reference]
            candidates = [str(ele) for ele in candidate]
            question =str(question)

            for candidate in candidates:
                for reference in references:
                    if reference not in confidece_scores:
                        confidece_scores[reference] = {}
                    confidece_scores[reference][candidate] = self.get_score(reference, candidate, question)
                        
            return confidece_scores
        elif isinstance(reference, list):
            references = [str(ele) for ele in reference]
            candidates =str(candidate)
            question =str(question)

            for reference in references:
                confidece_scores[reference] = {}
                confidece_scores[reference][candidate] = self.get_score(reference, candidate, question)
            
            return confidece_scores
        elif isinstance(candidate, list):
            candidates = [str(ele) for ele in candidate]
            reference =str(reference)
            question =str(question)

            if reference not in confidece_scores:
                confidece_scores[reference] = {}

            for candidate in candidates:
                confidece_scores[reference][candidate] = self.get_score(reference, candidate, question)
           
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
    Return True if the candidate answer is above the threshold value. Else, False.
    '''
    def transformer_match(self, reference, candidate, question, threshold=0.5):
        if em_match(reference, candidate) == True:
            return True
        judgment = False
        if isinstance(reference, list) and isinstance(candidate, list):
            candidates = [str(ele) for ele in candidate]
            references = [str(ele) for ele in reference]
            for candidate in candidates:
                if judgment == False:
                    for reference in references:
                        model_score = self.get_score(reference, candidate, question)
                        if model_score > threshold:
                            judgment = True
                            
            return judgment
        elif isinstance(reference, list):
            references = [str(ele) for ele in reference]
            candidate =str(candidate)
            for reference in references:
                model_score = self.get_score(reference, candidate, question)
                if model_score > threshold:
                    judgment = True
                    
            return judgment
        elif isinstance(candidate, list):
            candidates = [str(ele) for ele in candidate]
            reference =str(reference)
            for candidate in candidates:
                model_score = self.get_score(reference, candidate, question)
                if model_score > threshold:
                    judgment = True
                    
            return judgment
        else:
            model_score = self.get_score(reference, candidate, question)
            return True if model_score > threshold else False
