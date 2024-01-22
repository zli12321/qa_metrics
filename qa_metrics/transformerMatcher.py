import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import random
import numpy as np

class TransformerMatcher:
    def __init__(self, model='bert'):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        '''
        Fetch the model and tokenizer from the local directory
        '''
        current_dir = os.path.dirname(__file__)
        
        if model == 'bert':
            model_dir = os.path.join(current_dir, 'transformer_models/bert')
            
            # Ensure the target directory exists
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            self.model = BertForSequenceClassification.from_pretrained('Zongxia/answer_equivalence_bert', cache_dir=model_dir)
            self.tokenizer = BertTokenizer.from_pretrained('Zongxia/answer_equivalence_bert', cache_dir=model_dir)
            # self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            # self.config = BertConfig.from_pretrained(model_dir)
            # model = BertForSequenceClassification(config=self.config)
            # self.model.load_state_dict(torch.load(os.path.join(current_dir, 'transformer_models/bert', 'ae_tuned_bert.bin'), map_location=self.device))
            # self.model.to(self.device)

    def get_score(self, reference, candidate, question):
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
        bert_score = probabilities[1] if len(probabilities.shape) > 0 else probabilities

        return bert_score
    
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

    def transformer_match(self, reference, candidate, question, threshold=0.5):
        judgment = False
        if isinstance(reference, list) and isinstance(candidate, list):
            candidates = [str(ele) for ele in candidate]
            references = [str(ele) for ele in reference]
            for candidate in candidates:
                if judgment == False:
                    for reference in references:
                        bert_score = self.get_score(reference, candidate, question)
                        if bert_score > threshold:
                            judgment = True
                            
            return judgment
        elif isinstance(reference, list):
            references = [str(ele) for ele in reference]
            candidate =str(candidate)
            for reference in references:
                bert_score = self.get_score(reference, candidate, question)
                if bert_score > threshold:
                    judgment = True
                    
            return judgment
        elif isinstance(candidate, list):
            candidates = [str(ele) for ele in candidate]
            reference =str(reference)
            for candidate in candidates:
                bert_score = self.get_score(reference, candidate, question)
                if bert_score > threshold:
                    judgment = True
                    
            return judgment
        else:
            bert_score = self.get_score(reference, candidate, question)
            return True if bert_score > threshold else False
