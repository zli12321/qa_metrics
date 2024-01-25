# QA-Evaluation-Metrics

[![PyPI version qa-metrics](https://img.shields.io/pypi/v/qa-metrics.svg)](https://pypi.org/project/qa-metrics/) 


QA-Evaluation-Metrics is a fast and lightweight Python package for evaluating question-answering models. It provides various basic metrics to assess the performance of QA models. Check out our **CFMatcher**, a matching method going beyond token-level matching and is more efficient than LLM matchings but still retains competitive evaluation performance of transformer LLM models.

If you find this repo avialable, please cite our paper:
```bibtex
@misc{li2024cfmatch,
  title={CFMatch: Aligning Automated Answer Equivalence Evaluation with Expert Judgments For Open-Domain Question Answering}, 
  author={Zongxia Li and Ishani Mondal and Yijun Liang and Huy Nghiem and Jordan Boyd-Graber},
  year={2024},
  eprint={2401.13170},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

## Installation

To install the package, run the following command:

```bash
pip install qa-metrics
```

## Usage

The python package currently provides four QA evaluation metrics.

#### Exact Match
```python
from qa_metrics.em import em_match

reference_answer = ["Charles , Prince of Wales"]
candidate_answer = "Prince Charles"
match_result = em_match(reference_answer, candidate_answer)
print("Exact Match: ", match_result)
```

#### F1 Score
```python
from qa_metrics.f1 import f1_match,f1_score_with_precision_recall

f1_stats = f1_score_with_precision_recall(reference_answer[0], candidate_answer)
print("F1 stats: ", f1_stats)

match_result = f1_match(reference_answer, candidate_answer, threshold=0.5)
print("F1 Match: ", match_result)
```

#### CFMatch
```python
from qa_metrics.cfm import CFMatcher

question = "who will take the throne after the queen dies"
cfm = CFMatcher()
scores = cfm.get_scores(reference_answer, candidate_answer, question)
match_result = cfm.cf_match(reference_answer, candidate_answer, question)
print("Score: %s; CF Match: %s" % (scores, match_result))
```

#### Transformer Match
Our fine-tuned BERT model is on 🤗 [Huggingface](https://huggingface.co/Zongxia/answer_equivalence_bert?text=The+goal+of+life+is+%5BMASK%5D.). Our Package also supports downloading and matching directly. More Matching transformer models will be available 🔥🔥🔥

```python
from qa_metrics.transformerMatcher import TransformerMatcher

question = "who will take the throne after the queen dies"
tm = TransformerMatcher("bert")
scores = tm.get_scores(reference_answer, candidate_answer, question)
match_result = tm.transformer_match(reference_answer, candidate_answer, question)
print("Score: %s; CF Match: %s" % (scores, match_result))
```


## Updates
- [01/24/24] 🔥 The full paper is uploaded and can be accessed [here]([https://arxiv.org/abs/2310.14566](https://arxiv.org/abs/2401.13170)). The dataset is expanded and leaderboard is updated.
- Our Training Dataset is adapted and augmented from [Bulian et al](https://github.com/google-research-datasets/answer-equivalence-dataset). Our [dataset repo](https://github.com/zli12321/Answer_Equivalence_Dataset.git) includes the augmented training set and QA evaluation testing sets discussed in our paper. 

## License

This project is licensed under the [MIT License](LICENSE.md) - see the LICENSE file for details.

## Contact

For any additional questions or comments, please contact [zli12321@umd.edu].

