# QA-Evaluation-Metrics

[![PyPI version qa-metrics](https://img.shields.io/pypi/v/qa-metrics.svg)](https://pypi.org/project/qa-metrics/) 
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/17b7vrZqH0Yun2AJaOXydYZxr3cw20Ga6?usp=sharing)

QA-Evaluation-Metrics is a fast and lightweight Python package for evaluating question-answering models and prompting of black-box and open-source large language models. It provides various basic metrics to assess the performance of QA models. Check out our paper [**PANDA**](https://arxiv.org/abs/2402.11161), an efficient QA evaluation that retains competitive evaluation performance of transformer LLM models. 

### Updates
- Uopdated to version 0.2.8 
  - Supports prompting OPENAI GPT-series models and Claude Series models now. (Assuimg OPENAI version > 1.0)
  - Supports prompting various open source models such as LLaMA-2-70B-chat, LLaVA-1.5 etc by calling API from [deepinfra](https://deepinfra.com/models).


## Installation
* Python version >= 3.6
* openai version >= 1.0


To install the package, run the following command:

```bash
pip install qa-metrics
```

## Usage

The python package currently provides six QA evaluation methods.

#### Prompting LLM For Evaluation

Note: The prompting function can be used for any prompting purposes.

###### OpenAI
```python
from qa_metrics.prompt_llm import CloseLLM
model = CloseLLM()
model.set_openai_api_key(YOUR_OPENAI_KEY)
prompt = 'question: What is the Capital of France?\nreference: Paris\ncandidate: The capital is Paris\nIs the candidate answer correct based on the question and reference answer? Please only output correct or incorrect.'
model.prompt_gpt(prompt=prompt, model_engine='gpt-3.5-turbo', temperature=0.1, max_tokens=10)

'''
'correct'
'''
```

###### Anthropic
```python
model = CloseLLM()
model.set_anthropic_api_key(YOUR_Anthropic_KEY)
model.prompt_claude(prompt=prompt, model_engine='claude-v1', anthropic_version="2023-06-01", max_tokens_to_sample=100, temperature=0.7)

'''
'correct'
'''
```

###### deepinfra (See below for descriptions of more models)
```python
from qa_metrics.prompt_open_llm import OpenLLM
model = OpenLLM()
model.set_deepinfra_key(YOUR_DEEPINFRA_KEY)
model.prompt(message=prompt, model_engine='mistralai/Mixtral-8x7B-Instruct-v0.1', temperature=0.1, max_tokens=10)

'''
'correct'
'''
```

#### Exact Match
```python
from qa_metrics.em import em_match

reference_answer = ["The Frog Prince", "The Princess and the Frog"]
candidate_answer = "The movie \"The Princess and the Frog\" is loosely based off the Brother Grimm's \"Iron Henry\""
match_result = em_match(reference_answer, candidate_answer)
print("Exact Match: ", match_result)
'''
Exact Match:  False
'''
```

#### F1 Score
```python
from qa_metrics.f1 import f1_match,f1_score_with_precision_recall

f1_stats = f1_score_with_precision_recall(reference_answer[0], candidate_answer)
print("F1 stats: ", f1_stats)
'''
F1 stats:  {'f1': 0.25, 'precision': 0.6666666666666666, 'recall': 0.15384615384615385}
'''

match_result = f1_match(reference_answer, candidate_answer, threshold=0.5)
print("F1 Match: ", match_result)
'''
F1 Match:  False
'''
```

#### PANDA Match
```python
from qa_metrics.pedant import PEDANT

question = "Which movie is loosley based off the Brother Grimm's Iron Henry?"
pedant = PEDANT()
scores = pedant.get_scores(reference_answer, candidate_answer, question)
max_pair, highest_scores = pedant.get_highest_score(reference_answer, candidate_answer, question)
match_result = pedant.evaluate(reference_answer, candidate_answer, question)
print("Max Pair: %s; Highest Score: %s" % (max_pair, highest_scores))
print("Score: %s; PANDA Match: %s" % (scores, match_result))
'''
Max Pair: ('the princess and the frog', 'The movie "The Princess and the Frog" is loosely based off the Brother Grimm\'s "Iron Henry"'); Highest Score: 0.854451712151719
Score: {'the frog prince': {'The movie "The Princess and the Frog" is loosely based off the Brother Grimm\'s "Iron Henry"': 0.7131625951317375}, 'the princess and the frog': {'The movie "The Princess and the Frog" is loosely based off the Brother Grimm\'s "Iron Henry"': 0.854451712151719}}; PANDA Match: True
'''
```

```python
print(pedant.get_score(reference_answer[1], candidate_answer, question))
'''
0.7122460127464126
'''
```

#### Transformer Match
Our fine-tuned BERT model is on 🤗 [Huggingface](https://huggingface.co/Zongxia/answer_equivalence_bert?text=The+goal+of+life+is+%5BMASK%5D.). Our Package also supports downloading and matching directly. [distilroberta](https://huggingface.co/Zongxia/answer_equivalence_distilroberta), [distilbert](https://huggingface.co/Zongxia/answer_equivalence_distilbert), [roberta](https://huggingface.co/Zongxia/answer_equivalence_roberta), and [roberta-large](https://huggingface.co/Zongxia/answer_equivalence_roberta-large) are also supported now! 🔥🔥🔥

```python
from qa_metrics.transformerMatcher import TransformerMatcher

question = "Which movie is loosley based off the Brother Grimm's Iron Henry?"
tm = TransformerMatcher("bert")
scores = tm.get_scores(reference_answer, candidate_answer, question)
match_result = tm.transformer_match(reference_answer, candidate_answer, question)
print("Score: %s; bert Match: %s" % (scores, match_result))
'''
Score: {'The Frog Prince': {'The movie "The Princess and the Frog" is loosely based off the Brother Grimm\'s "Iron Henry"': 0.6934309}, 'The Princess and the Frog': {'The movie "The Princess and the Frog" is loosely based off the Brother Grimm\'s "Iron Henry"': 0.7400551}}; TM Match: True
'''
```

If you find this repo avialable, please cite:
```bibtex
@misc{li2024panda,
      title={PANDA (Pedantic ANswer-correctness Determination and Adjudication):Improving Automatic Evaluation for Question Answering and Text Generation}, 
      author={Zongxia Li and Ishani Mondal and Yijun Liang and Huy Nghiem and Jordan Lee Boyd-Graber},
      year={2024},
      eprint={2402.11161},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Updates
- Improved PANDA evaluation with more representative answer correctness training data.
- 🔥 The full paper is uploaded and can be accessed [here](https://arxiv.org/abs/2402.11161). The dataset is expanded and leaderboard is updated.
- Our Training Dataset is adapted and augmented from [Bulian et al](https://github.com/google-research-datasets/answer-equivalence-dataset). Our [dataset repo](https://github.com/zli12321/Answer_Equivalence_Dataset.git) includes the augmented training set and QA evaluation testing sets discussed in our paper.
- Now our model supports [distilroberta](https://huggingface.co/Zongxia/answer_equivalence_distilroberta), [distilbert](https://huggingface.co/Zongxia/answer_equivalence_distilbert), a smaller and faster matching model than Bert!
- Now our model supports [roberta](https://huggingface.co/Zongxia/answer_equivalence_roberta), [roberta-large](https://huggingface.co/Zongxia/answer_equivalence_roberta-large), a larger and more robust matching model than Bert!
- Check avilability of open-source LLMs in [deepinfra](https://deepinfra.com/models)
- deepinfra supports: "lizpreciatior/lzlv_70b_fp16_hf", "meta-llama/Llama-2-70b-chat-hf", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "01-ai/Yi-34B-Chat", "google/gemma-7b-it", "llava-hf/llava-1.5-7b-hf", "mistralai/Mixtral-8x7B-Instruct-v0.1"

## License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE file for details.

## Contact

For any additional questions or comments, please contact [zli12321@umd.edu].

