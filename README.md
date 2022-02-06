# Claim detection model comparison
---
![GitHub Pipenv locked Python version](https://img.shields.io/github/pipenv/locked/python-version/jueri/claim_model_comparison)
### 💡 Info:
This repository holds the code for a sentence based claim detection model comparison. The main modules can be found in the [src](https://github.com/jueri/claim_model_comparison/src) directory. The notebooks in the root directory interface the claim detection models and guide through the training and evaluation process.

This repository is part of my bachelore theses with the title **Automated statement extractionfrom press briefings**. For more indepth information see the [Statement Extractor](https://github.com/jueri/statement_extractor) repository.

### ⚙️ Setup:
This repository uses Pipenv to manage a virtual environment with all python packages. Information about how to install Pipenv can be found [here](https://pipenv.pypa.io/en/latest/).
To create a virtual environment and install all packages needed, call `pipenv install` from the root directory.

Additionally, a current JDK version is needed to run the CSQ system since it is based on [Pyserini](https://github.com/castorini/pyserini/) based on [Anserini](https://github.com/castorini/anserini). The system was tested with `openjdk-11-jdk`.

The [transformers](https://github.com/huggingface/transformers) library used for the BERT models needs a current [Rust](https://www.rust-lang.org/) installation.


Default directorys and parameter can be defined in [config.py](https://github.com/jueri/claim_model_comparison/tree/master/config.py).
After installation please run the [setup.py](https://github.com/jueri/claim_model_comparison/tree/master/setup.py) file to create the expected directories and download additional data.


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
└── IBM_Debater_(R)_claim_sentences_search          # DATASET_2018
    ├── claim_sentence_search.csv
    ├── q_mc_heldout.csv
    ├── q_mc_test.csv
    ├── q_mc_train.csv
    ├── readme_mc_queries.txt
    ├── readme_test_set.txt
    └── test_set.csv 
```
