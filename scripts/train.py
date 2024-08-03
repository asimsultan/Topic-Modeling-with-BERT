
import os
import torch
import argparse
import pandas as pd
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from utils import preprocess_text

def main(data_path):
    # Parameters
    model_name = 'bert-base-uncased'
    num_topics = 10
    max_length = 128

    # Load Dataset
    dataset = pd.read_csv(data_path)
    texts = dataset['text'].tolist()

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Preprocess Data
    tokenized_texts = [preprocess_text(tokenizer, text, max_length) for text in texts]

    # BERT Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Extract BERT embeddings
    embeddings = []
    with torch.no_grad():
        for tokenized_text in tokenized_texts:
            inputs = torch.tensor(tokenized_text).unsqueeze(0).to(device)
            outputs = model(inputs)
            embeddings.append(outputs.last_hidden_state.mean(dim=1).cpu().numpy())

    embeddings = np.vstack(embeddings)

    # Vectorizer and LDA Model
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=0)

    # Fit LDA Model
    lda_model.fit(vectorizer.fit_transform(texts))

    # Save Models
    model_dir = './models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(model.state_dict(), os.path.join(model_dir, 'bert.pth'))
    with open(os.path.join(model_dir, 'lda.pkl'), 'wb') as f:
        pickle.dump(lda_model, f)
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing text data')
    args = parser.parse_args()
    main(args.data_path)
