
import torch
import argparse
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from utils import preprocess_text

def main(model_path, data_path):
    # Load Models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained('bert-base-uncased')
    model.load_state_dict(torch.load(os.path.join(model_path, 'bert.pth')))
    model.to(device)
    model.eval()

    with open(os.path.join(model_path, 'lda.pkl'), 'rb') as f:
        lda_model = pickle.load(f)
    with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    # Load Dataset
    dataset = pd.read_csv(data_path)
    texts = dataset['text'].tolist()

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Preprocess Data
    tokenized_texts = [preprocess_text(tokenizer, text, 128) for text in texts]

    # Extract BERT embeddings
    embeddings = []
    with torch.no_grad():
        for tokenized_text in tokenized_texts:
            inputs = torch.tensor(tokenized_text).unsqueeze(0).to(device)
            outputs = model(inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    embeddings = np.vstack(embeddings)

    # Transform texts with vectorizer and predict topics
    transformed_texts = vectorizer.transform(texts)
    topic_distributions = lda_model.transform(transformed_texts)

    # Print topics
    for idx, topic_distribution in enumerate(topic_distributions):
        print(f'Text: {texts[idx]}')
        print(f'Topics: {topic_distribution}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the directory containing the trained models')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing text data')
    args = parser.parse_args()
    main(args.model_path, args.data_path)
