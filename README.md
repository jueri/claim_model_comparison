# Claim detection model comparison
### Info 💡
This repository holds the code for a sentence based claim detection model comparison. The main modules can be found in the [src](https://github.com/jueri/claim_model_comparison/src) directory. The notebooks in the root directory interface the claim detection models and guide through the training and evaluation process.

This repository is part of my bachelore theses with the title **Automated statement extractionfrom press briefings**. For more indepth information see the [Statement Extractor](https://github.com/jueri/statement_extractor) repository.

### Setup 🎛
This repo holds a [Visual Studio Code (VS Code)](https://code.visualstudio.com/) [.devcontainer](https://github.com/jueri/SMC_claim_dataset/tree/master/.devcontainer). The docker development container can easily be recreated using VS Code.
Alternatively, can the dependencies be installed using with the following command:
`pip install -r .devcontainer/requirements.txt`

Default directorys and parameter can be defined in [config.py](https://github.com/jueri/claim_model_comparison/tree/master/config.py).
After installation please run the [setup.py](https://github.com/jueri/claim_model_comparison/tree/master/setup.py) file to create the expected directories and download additional data.

### Content 📋
Results:
- [dataset_analysis.ipynb](https://github.com/jueri/claim_model_comparison/tree/master/dataset_analysis.ipynb) holds the analysis of the results.

Models:
- [claim_model_BERT](https://github.com/jueri/claim_model_comparison/tree/master/claim_model_BERT.ipynb)
- [claim_model_CSQ](https://github.com/jueri/claim_model_comparison/tree/master/claim_model_CSQ.ipynb)
- [claim_model_fasttext](https://github.com/jueri/claim_model_comparison/tree/master/claim_model_fasttext.ipynb)
- [claim_model_LogisticRegression](https://github.com/jueri/claim_model_comparison/tree/master/claim_model_LogisticRegression.ipynb)
- [claim_model_LSTM](https://github.com/jueri/claim_model_comparison/tree/master/claim_model_LSTM.ipynb)
- [claim_model_SVM](https://github.com/jueri/claim_model_comparison/tree/master/claim_model_SVM.ipynb)


### Data 💾
The models are trained on two datasets: *IBM Debater® - Claims and Evidence* (`IBM_Debater_(R)_CE-ACL-2014.v0`) and *IBM Debater® - Claim Sentences Search* (`IBM_Debater_(R)_claim_sentences_search`). The datasets can be downloaded [here](https://research.ibm.com/haifa/dept/vst/debating_data.shtml) and are expected in the data directory: 
```
├── IBM_Debater_(R)_CE-ACL-2014.v0                  # DATASET_2014
│   ├── 2014_7_18_ibm_CDCdata.xls
│   ├── 2014_7_18_ibm_CDEdata.xls
│   ├── CE-ACL_processed.csv
│   ├── ReadMe.txt
│   ├── wiki12_articles
│   └── ...
├── IBM_Debater_(R)_claim_sentences_search          # DATASET_2018
│   ├── claim_sentence_search.csv
│   ├── q_mc_heldout.csv
│   ├── q_mc_test.csv
│   ├── q_mc_train.csv
│   ├── readme_mc_queries.txt
│   ├── readme_test_set.txt
│   └── test_set.csv
└── wandb_export_2021-12-22T15 30 01.504+01 00.csv  # W&B results export
```