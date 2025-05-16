import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def sentence_splitter(text):
    return [s.strip() for s in sent_tokenize(text) if s.strip()]

class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

def predict_sentence_scores(sentences, model, tokenizer, batch_size=32):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = SentenceDataset(sentences)
    loader = DataLoader(dataset, batch_size=batch_size)

    all_scores = []

    with torch.no_grad():

        for batch in tqdm(loader):
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=1)
            p1 = probs[:, 1].cpu().tolist()
            all_scores.extend(p1)

    return all_scores

def compute_labels(sentences, scores):
    mean_p1 = sum(scores) / len(scores)
    global_label = '1' if mean_p1 >= 0.5 else '0'
    sentence_level = list(zip(sentences, scores))
    return sentence_level, mean_p1, global_label

def processing(df, model_name = 'model', title_col='Title', abstract_col='text'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token = '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')
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token ='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').to(device)
    model.eval()

    #1
    df = df.copy() #copie du dataset pour le subset
    df['new_text'] = df[title_col].fillna('').astype(str) + ' ' + df[abstract_col].fillna('').astype(str)
    #on joint le titre et l'abstract sans utiliser de point (le titre a dejà un point)

    #2
    df['sentences'] = df['new_text'].apply(sentence_splitter) #les blocs de texte sont split en phrases dans une liste dans une nouvelle colonne
    all_sentences = [s for sublist in df['sentences'] for s in sublist] #toutes les phrases sont mises simultanéments dans un grand fichier unique

    # 3
    print("Running model inference...")
    all_scores = predict_sentence_scores(all_sentences, model, tokenizer) #les phrases sont transformées en CustomDataset HF puis sont injectées par batch dans le modèle

    # 4
    sentence_lengths = df['sentences'].apply(len).tolist() #On extrait dans une liste la longueur (en phrase de chaque bloc de texte initial)
    pointer = 0
    outputs = []
    for i, slen in enumerate(sentence_lengths):
        sent = df.at[i, 'sentences']
        scores = all_scores[pointer:pointer + slen]
        pointer += slen
        outputs.append(compute_labels(sent, scores))

    df[['sentence_level_predictions',
        'probability_positive_class',
        'normal_papermill_probability']] = pd.DataFrame(outputs, index=df.index)

    return df

if __name__ == '__main__':
    start = time.time()
    df = pd.read_excel('/home/bscancar/cdd_institut_agro/df_bert_inference_comparison.xlsx')
    df = processing(df)
    df.to_excel('/home/bscancar/cdd_institut_agro/df_bert_inference_comparison_opti.xlsx', index=False)
    print(f"✅ Done in {time.time()-start:.2f}s")
