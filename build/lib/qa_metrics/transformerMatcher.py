import os
import torch
from .utils.tools import normalize_answer
from transformers import BertForSequenceClassification, BertTokenizer

class TransformerMatcher:
    def __init__(self, model='bert'):
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
        input_text = "[CLS] " + normalize_answer(str(candidate)) + " [SEP] " + normalize_answer(str(reference)) + " [SEP] " + normalize_answer(question) + " [SEP]"
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
    
    def transformer_match(self, reference, candidate, question, threshold=0.5):
        bert_score = self.get_score(reference, candidate, question)
        binary_class = True if bert_score > threshold else False
        return binary_class
