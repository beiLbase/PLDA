
## Few-Shot Relation Extraction with Prototypical Network Combing Prompt Learning and Data Augmentation (PLDA)


## TASK
- <span style="font-size: 20px; font-weight: bold;">designing a query-aware prototypical network learning</span>
- <span style="font-size: 20px; font-weight: bold;">integrating query-aware prototypical network with relational information</span>





##  Acknowledgements

This repository contains the implementation of PLDA, a few-shot relation extraction model developed as part of my doctor's dissertation. The model constructs multiple structured prompt templates based on entity type constraints and formalizes the relation extraction task into a masked language modeling task, thereby guiding the classification model to incorporate general knowledge from the pre-trained language model. Furthermore, this method utilizes the GPT-3.5 large language model to generate semantically consistent and sentence-structure-diverse augmented instances to expand the support set and alleviate class-prototype bias in few-shot scenarios. The related works are listed below in acknowledgment of their contributions.


## Related Works

FewRel: A Large-Scale Supervised Few-Sample Relation Classification Dataset with State-of-the-Art Evaluation Results 

By Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, Maosong Sun. EMNLP 2018.

DOI: https://doi.org/10.18653/v1/D18-1514

The dataset: https://github.com/thunlp/FewRel


KnowPrompt: Knowledge-aware Prompt-tuning with Synergistic Optimization for Relation Extraction

By Xiang Chen, Ningyu Zhang, Xin Xie, Shumin Deng, Yunzhi Yao, Chuanqi Tan, Fei Huang, Luo Si, Huajun Chen. WWW 2022.

DOI: https://github.com/zjunlp/KnowPrompt


