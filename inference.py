import pandas as pd
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from bs4 import BeautifulSoup

def clean_structured_abstract(text):

    if not isinstance(text, str):
        return text 

    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")

    headers = [
        "background", "objective", "objectives", "methods", "materials and methods",
        "results", "conclusion", "conclusions", "introduction", "design",
        "setting", "participants", "main outcome measures", "background and aim",
        "abstract", "abstract aims", "purpose", "aim", "aims", "Study Hypothesis", "Sample Size",
        "Exclusion criteria", "Primary Objective", "Major Inclusion/Exclusion Criteria Inclusion criteria",
        "Trial Registration", "Eligibility Criteria", "Data extraction and synthesis", "Primary and secondary outcome measures",
        "Interventions", "Background/Objectives", "Introduction", "Methods and analysis", "Ethics and dissemination", "purose",
        "Importance", "Data sources and study selection"
    ]

    headers.sort(key=lambda x: -len(x)) 
    header_group = "|".join(map(re.escape, headers))

    pattern = r"(?imx)^ \s* (?:{}) \s* (?:[:.\n\r]+|(?=\s*[A-Z]))".format(header_group)
    text = re.sub(pattern, "", text)

    inline_pattern = r"(?i)([.;,:!?])\s+(?:{})\s*[:.]".format(header_group)
    text = re.sub(inline_pattern, r"\1", text)

    pattern_direct_caps = r"(?i)\b(?:{})\b(?=\s+[A-Z])".format(header_group)
    text = re.sub(pattern_direct_caps, "", text)

    cleaned_text = re.sub(r"\s+", " ", text).strip()

    return cleaned_text



def sentence_splitter(text):
    
    try:
        new_text = clean_structured_abstract(text)
        protected = re.sub(r'\bet al\.', 'et al§', new_text)
        return [s.strip().replace('et al§', 'et al.') for s in sent_tokenize(protected) if s.strip()]
    
    except LookupError:
        nltk.download("punkt")
        nltk.download('punkt_tab')
        return [s.strip().replace('et al§', 'et al.') for s in sent_tokenize(protected) if s.strip()]



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

def document_processing(df, model_name = 'model', title_col='Title', abstract_col='text'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
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
        'papermill_probability',
        'prediction']] = pd.DataFrame(outputs, index=df.index)

    return df


def pipeline_single_text(sentences, model_name) :
    my_pipeline = pipeline(model=model_name,
             top_k = 1,
             )
    results = my_pipeline(sentences)
    probs = [r[0]['score'] if r[0]['label'] == 'LABEL_1' else 1-r[0]['score']
             for r in results ]
    
    mean_prob = sum(probs) / len(probs)
    label = 1 if mean_prob >= 0.5 else 0

    ## results
    df = pd.DataFrame(sentences, columns=['sentences'])

    all_probabilities = probs
    all_labels = [1 if a >= 0.5 else 0 for a in all_probabilities]

    df[['papermill_probability', 'label']] = list(zip(all_probabilities, all_labels))

    return mean_prob, label, df

def text_processing(text, model_name) :
    sentences = sentence_splitter(text)
    papermill_probability, label, df = pipeline_single_text(sentences, model_name)
    
    return papermill_probability, df



if __name__ == '__main__':
    start = time.time()
    df = pd.read_excel('/home/bscancar/cdd_institut_agro/df_bert_inference_comparison.xlsx')
    df = document_processing(df)
    df.to_excel('/home/bscancar/cdd_institut_agro/df_bert_inference_comparison_opti.xlsx', index=False)
    print(f"✅ Done in {time.time()-start:.2f}s")
