# QA-Evaluation-Metrics 📊

[![PyPI version qa-metrics](https://img.shields.io/pypi/v/qa-metrics.svg)](https://pypi.org/project/qa-metrics/) 
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ke23KIeHFdPWad0BModmcWKZ6jSbF5nI?usp=sharing)

> A fast and lightweight Python package for evaluating question-answering models and prompting of black-box and open-source large language models.

## 🎉 Latest Updates

- **Version 0.2.19 Released!**
  - Paper accepted to EMNLP 2024 Findings! 🎓
  - Enhanced PEDANTS with multi-pipeline support and improved edge case handling
  - Added support for OpenAI GPT-series and Claude Series models (OpenAI version > 1.0)
  - Integrated support for open-source models (LLaMA-2-70B-chat, LLaVA-1.5, etc.) via [deepinfra](https://deepinfra.com/models)
  - Introduced trained tiny-bert for QA evaluation (18MB model size)
  - Added direct Huggingface model download support for TransformerMatcher

## 🚀 Quick Start

### Prerequisites
- Python >= 3.6
- openai >= 1.0

### Installation
```bash
pip install qa-metrics
```

## 💡 Features

Our package offers six QA evaluation methods with varying strengths:

| Method | Best For | Cost | Correlation with Human Judgment |
|--------|----------|------|--------------------------------|
| Normalized Exact Match | Short-form QA (NQ-OPEN, HotpotQA, etc.) | Free | Good |
| PEDANTS | Both short & medium-form QA | Free | Very High |
| [Neural Evaluation](https://huggingface.co/zli12321/answer_equivalence_tiny_bert) | Both short & long-form QA | Free | High |
| [Open Source LLM Evaluation](https://huggingface.co/zli12321/prometheus2-2B) | All QA types | Free | High |
| Black-box LLM Evaluation | All QA types | Paid | Highest |

## 📖 Documentation

### 1. Normalized Exact Match

#### Method: `em_match`
**Parameters**
- `reference_answer` (list of str): A list of gold (correct) answers to the question
- `candidate_answer` (str): The answer provided by a candidate that needs to be evaluated

**Returns**
- `boolean`: True if there are any exact normalized matches between gold and candidate answers

```python
from qa_metrics.em import em_match

reference_answer = ["The Frog Prince", "The Princess and the Frog"]
candidate_answer = "The movie \"The Princess and the Frog\" is loosely based off the Brother Grimm's \"Iron Henry\""
match_result = em_match(reference_answer, candidate_answer)
```

### 2. F1 Score

#### Method: `f1_score_with_precision_recall`
**Parameters**
- `reference_answer` (str): A gold (correct) answer to the question
- `candidate_answer` (str): The answer provided by a candidate that needs to be evaluated

**Returns**
- `dictionary`: Contains the F1 score, precision, and recall between a gold and candidate answer

#### Method: `f1_match`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `candidate_answer` (str): Candidate answer to evaluate
- `threshold` (float): F1 score threshold for considering a match (default: 0.5)

**Returns**
- `boolean`: True if F1 score exceeds threshold for any gold answer

```python
from qa_metrics.f1 import f1_match, f1_score_with_precision_recall

f1_stats = f1_score_with_precision_recall(reference_answer[0], candidate_answer)
match_result = f1_match(reference_answer, candidate_answer, threshold=0.5)
```

### 3. PEDANTS

#### Method: `get_score`
**Parameters**
- `reference_answer` (str): A Gold answer
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `float`: The similarity score between two strings (0 to 1)

#### Method: `get_highest_score`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `dictionary`: Contains the gold answer and candidate answer pair with highest matching score

#### Method: `get_scores`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `dictionary`: Contains matching scores for all gold answer and candidate answer pairs

#### Method: `evaluate`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `boolean`: True if candidate answer matches any gold answer

#### Method: `get_question_type`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `question` (str): The question being evaluated

**Returns**
- `list`: The type of the question (what, who, when, how, why, which, where)

#### Method: `get_judgement_type`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `list`: A list revised rules applicable to judge answer correctness

```python
from qa_metrics.pedant import PEDANT

pedant = PEDANT()
scores = pedant.get_scores(reference_answer, candidate_answer, question)
match_result = pedant.evaluate(reference_answer, candidate_answer, question)
```

### 4. Transformer Neural Evaluation

#### Method: `get_score`
**Parameters**
- `reference_answer` (str): A Gold answer
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `float`: The similarity score between two strings (0 to 1)

#### Method: `get_highest_score`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `dictionary`: Contains the gold answer and candidate answer pair with highest matching score

#### Method: `get_scores`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `dictionary`: Contains matching scores for all gold answer and candidate answer pairs

#### Method: `transformer_match`
**Parameters**
- `reference_answer` (list of str): List of gold answers
- `candidate_answer` (str): Candidate answer to evaluate
- `question` (str): The question being evaluated

**Returns**
- `boolean`: True if transformer model considers candidate answer equivalent to any gold answer

```python
from qa_metrics.transformerMatcher import TransformerMatcher

### supports `zli12321/answer_equivalence_bert`, `zli12321/answer_equivalence_distilbert`, `zli12321/answer_equivalence_roberta`, `zli12321/answer_equivalence_distilroberta`
tm = TransformerMatcher("zli12321/answer_equivalence_tiny_bert")
match_result = tm.transformer_match(reference_answer, candidate_answer, question)
```

### 5. LLM Integration

#### Method: `prompt_gpt`
**Parameters**
- `prompt` (str): The input prompt text
- `model_engine` (str): OpenAI model to use (e.g., 'gpt-3.5-turbo')
- `temperature` (float): Controls randomness (0-1)
- `max_tokens` (int): Maximum tokens in response

```python
from qa_metrics.prompt_llm import CloseLLM

model = CloseLLM()
model.set_openai_api_key(YOUR_OPENAI_KEY)
result = model.prompt_gpt(prompt=prompt, model_engine='gpt-3.5-turbo')
```

#### Method: `prompt_claude`
**Parameters**
- `prompt` (str): The input prompt text
- `model_engine` (str): Claude model to use
- `anthropic_version` (str): API version
- `max_tokens_to_sample` (int): Maximum tokens in response
- `temperature` (float): Controls randomness (0-1)

```python
model = CloseLLM()
model.set_anthropic_api_key(YOUR_ANTHROPIC_KEY)
result = model.prompt_claude(prompt=prompt, model_engine='claude-v1')
```

#### Method: `prompt`
**Parameters**
- `message` (str): The input message text
- `model_engine` (str): Model to use
- `temperature` (float): Controls randomness (0-1)
- `max_tokens` (int): Maximum tokens in response

```python
from qa_metrics.prompt_open_llm import OpenLLM

model = OpenLLM()
model.set_deepinfra_key(YOUR_DEEPINFRA_KEY)
result = model.prompt(message=prompt, model_engine='mistralai/Mixtral-8x7B-Instruct-v0.1')
```

## 🤗 Model Hub

Our fine-tuned models are available on Huggingface:
- [BERT](https://huggingface.co/Zongxia/answer_equivalence_bert)
- [DistilRoBERTa](https://huggingface.co/Zongxia/answer_equivalence_distilroberta)
- [DistilBERT](https://huggingface.co/Zongxia/answer_equivalence_distilbert)
- [RoBERTa](https://huggingface.co/Zongxia/answer_equivalence_roberta)
- [Tiny-BERT](https://huggingface.co/Zongxia/answer_equivalence_tiny_bert)
- [RoBERTa-Large](https://huggingface.co/Zongxia/answer_equivalence_roberta-large)

## 📚 Resources

- [Full Paper](https://arxiv.org/abs/2402.11161)
- [Dataset Repository](https://github.com/zli12321/Answer_Equivalence_Dataset.git)
- [Supported Models on Deepinfra](https://deepinfra.com/models)

## 📄 Citation

```bibtex
@misc{li2024pedantspreciseevaluationsdiverse,
      title={PEDANTS: Cheap but Effective and Interpretable Answer Equivalence}, 
      author={Zongxia Li and Ishani Mondal and Yijun Liang and Huy Nghiem and Jordan Lee Boyd-Graber},
      year={2024},
      eprint={2402.11161},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.11161}, 
}
```

## 📝 License

This project is licensed under the [MIT License](LICENSE.md).

## 📬 Contact

For questions or comments, please contact: zli12321@umd.edu
