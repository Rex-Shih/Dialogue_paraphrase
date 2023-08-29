from evaluate import load
import json
import numpy as np
import random
from transformers import AutoTokenizer, RobertaForSequenceClassification
import argparse
from tqdm.contrib import tzip
parser = argparse.ArgumentParser()
parser.add_argument("--predicted_file", type=str)
args = parser.parse_args()

seed = 314
random.seed(seed)
np.random.seed(seed)


refs = []
predictions = []
prev = []
future=[]

ver2_predictions = []
ver2_refs = []
ver2_predictions = []
ver2_prev = []
ver2_future=[]

sacrebleu = load('sacrebleu')
rouge=load('rouge')
bert=load('bertscore')
with open('./script_data/testData.json') as f:
    for i in f:
        json_obj = json.loads(i)
        prev.append(json_obj['prev'])
        refs.append(json_obj['cur'])
        future.append(json_obj['future'])
    f.close()

with open('./script_data/testData_withHint.json') as f:
    for i in f:
        json_obj = json.loads(i)
        ver2_prev.append(json_obj['prev'])
        ver2_refs.append(""+ver2_prev[-1][-2:]+json_obj['cur'][:-1])
        ver2_future.append(json_obj['future'])
    f.close()
with open(args.predicted_file) as f:
    for num, i in enumerate(f):
        if i[-1] == '\n':
            predictions.append(i[:-1])
            ver2_predictions.append(""+ver2_prev[num][-2:]+' '+i[:-1])
        else:
            predictions.append(i)
            ver2_predictions.append(""+ver2_prev[num][-2:]+' '+i)
        ver2_prev[num] = ver2_prev[num][:-3]
        
    f.close()

bleu_res = sacrebleu.compute(predictions=predictions, references=refs)
rouge_res = rouge.compute(predictions=predictions, references=refs)
results = bert.compute(predictions=predictions, references=refs, model_type="bert-base-uncased")
bert_res = sum(results['f1'])/len(results['f1'])

#initialize model

tokenizer2 = AutoTokenizer.from_pretrained("roberta-large")
model_ver2 = RobertaForSequenceClassification.from_pretrained("../train_model/model/bestNSP")
model_ver2.to('cuda')

def _score_ver2(a, b):
        encoded = tokenizer2.encode_plus(a, text_pair=b, return_tensors='pt', max_length=512)
        for k in encoded:
            encoded[k] = encoded[k].to('cuda')
        seq_relationship_logits = model_ver2(**encoded)[0]
        return (seq_relationship_logits[0, 0].tolist())

nsp_front=0
nsp_back=0

for data, p, f in tzip(ver2_predictions, ver2_prev, ver2_future):
    fst=_score_ver2(p, data+' '+f)
    snd = _score_ver2(p+' '+data, f)
    nsp_front+=fst
    nsp_back+=snd
    
print("bleu = : "+str(round(bleu_res['score'], 2)))
print("rouge-l =: "+str(round(rouge_res['rougeL']*100, 2)))
print("bertscore = :"+str(round(bert_res*100, 2)))
print("nsp score 1 = "+str(round(nsp_front/len(ver2_refs), 2)))
print("nsp score 2= "+str(round(nsp_back/len(ver2_refs), 2)))
print("* Note that nsp score 1 is the nsp score between previous sentences and prediction + future sentences; While nsp score 2 is the nsp score between previous sentences + prefiction and future sentences.")

