
# Topic Modeling with BERT

Welcome to the Topic Modeling with BERT project! This project focuses on extracting topics from text data using BERT embeddings and Latent Dirichlet Allocation (LDA).

## Introduction

Topic modeling involves identifying the underlying topics present in a collection of documents. In this project, we leverage the power of BERT to generate embeddings for text data and use LDA to extract topics.

## Dataset

For this project, we will use a custom dataset of text data. You can create your own dataset and place it in the `data/topic_modeling_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Pandas
- Scikit-learn

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/bert_topic_modeling.git
cd bert_topic_modeling

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes text data. Place these files in the data/ directory.
# The data should be in a CSV file with one column: text.

# To train the BERT and LDA models for topic modeling, run the following command:
python scripts/train.py --data_path data/topic_modeling_data.csv

# To evaluate the performance of the trained models, run:
python scripts/evaluate.py --model_path models/ --data_path data/topic_modeling_data.csv
