from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, GPTNeoForCausalLM
import argparse
from operator import add
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import csv
from tqdm import tqdm
import random
import transformers
import torch.nn.functional as F
import evaluate
def get_txt_dialogs(train_addr, valid_addr, test_addr):
    def splitting(file):
        arr = []
        for txt in file:
            sent = txt.split(" __eou__ ")
            sent[-1] = sent[-1].split("__eou__\n")[0]
            arr.append(sent)
        return arr
    train = open(train_addr, 'r')
    valid = open(valid_addr, 'r')
    test = open(test_addr, 'r')
    return splitting(train), splitting(valid), splitting(test)    
def get_jsn_dialog(train_addr, valid_addr, test_addr):
    def splitting(file):
        dialog = []
        for i in file:
            tmp = json.loads(i)
            arr=[]
            for j in tmp['dialog']:
                arr.append(j['text'])
            dialog.append(arr)
        #print(len(dialog))
        return dialog
    train = open(train_addr, 'r')
    valid = open(valid_addr, 'r')
    test = open(test_addr, 'r')
    return splitting(train), splitting(valid), splitting(test)

def get_wow_dialog(train_addr, valid_addr, test_addr):
    def splitting(file):
        dialog = []
        for  i in file:
            i = json.loads(i)
            for j in i:
                tmp = []
                for sent in j['dialog']:
                    tmp.append(sent['text'])
                dialog.append(tmp)
        #print(len(dialog))
        return dialog
    train = open(train_addr, 'r')
    valid = open(valid_addr, 'r')
    test = open(test_addr, 'r')
    return splitting(train), splitting(valid), splitting(test)
def data_processing(train, valid, test, tokenizer):
    def proc_txt(file):
        ret = []
        max = 0
        for tmp_idx, sent in enumerate(file):
            encoded_sent = ""
            for i, raw_sent in enumerate(sent):
                if i%2 == 0:
                    heading = "A: "
                else:
                    heading = "B: "
                encoded_sent = encoded_sent+' '+heading+raw_sent
            #for i in sent:
            
            tmp = tokenizer(encoded_sent[1:], return_tensors='pt', max_length=512, truncation=True, padding='max_length')
               
            tmp['labels'] = tmp['input_ids']
            ret.append(tmp)
            """accum = sent[0]
            for i in range(1, len(sent)):
                accum += " "+sent[i]
            tmp = tokenizer(accum, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
            tmp['labels'] = tmp['input_ids']
            ret.append(tmp)"""
        return ret
    
    return  proc_txt(train),  proc_txt(valid),  proc_txt(test)
class DailyDialogueDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return self.encodings[idx]
    def __len__(self):
        return len(self.encodings)
def compute_metrics(eval_predictions):
   
    label_ids = eval_predictions.label_ids
    preds = eval_predictions.predictions[0]
    preds = np.squeeze(preds, axis=1)
    label_ids = np.squeeze(label_ids, axis=1)
    sent_pred = []
    sent_ref = []
    lens = preds.shape[0]
   
    #losss= 0
    #losss = F.cross_entropy(torch.tensor(predictions[0][:-1]), torch.tensor(label_ids[0][1:]))
    for i in range(lens):
        for j in range(512):
            if label_ids[0][j]==50256:
                break
        sent_pred.append(tokenizer.decode(preds[i][:j-1]))#, skip_special_tokens=True))
        #print(sent_pred[-1])
        sent_ref.append(tokenizer.decode(label_ids[i][1:j]))
    
    bleu_results = sacrebleu.compute(predictions=sent_pred, references=sent_ref)
    rouge_res = rouge.compute(predictions=sent_pred, references=sent_ref)
    results = bert.compute(predictions=sent_pred, references=sent_ref, model_type="bert-base-uncased")
    bert_res = sum(results['f1'])/len(results['f1'])
    return {"bleu score": bleu_results['score'], "rouge score":str(rouge_res['rougeL']*100), "bert score": str(bert_res*100)}

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
   
    pred_ids = torch.argmax(logits, dim=-1)
    
    return pred_ids, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model", type=str, default="gpt2-medium",
        help="pretrained model name or path to local checkpoint")
    parser.add_argument("--epoch", type=int, default=10,help="Training epoch.")
    parser.add_argument("--batch_size", type=int, default=8,help="Training batch size")
    parser.add_argument("--train_addr", type=str, default="../../data/EMNLP_dataset/train/dialogues_train.txt", help="train dataset")
    parser.add_argument("--valid_addr", type=str, default="../../data/EMNLP_dataset/validation/dialogues_validation.txt", help="evaluation dataset")
    parser.add_argument("--test_addr", type=str, default="../../data/EMNLP_dataset/test/dialogues_test.txt", help="test dataset")
    parser.add_argument("--lr", type=float, default=5e-6, help="training learning rate")
    parser.add_argument("--dropout", type=float, default=0.2, help="training dropout")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate states to help with tuning / debugging.")
    parser.add_argument("--output_dir", type=str, default="../model/", help="Output dir.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true", help="fp16")
    parser.add_argument("--data_seed", type=int, default=3141, help="seed for data")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="gradient accumulation steps")
    parser.add_argument("--test", action="store_true", help="whether to train the model or test the model")
    parser.add_argument("--pretrain", action="store_true", help="whether to use wow and msc datasets to pretrain the model")
    args = parser.parse_args()
    #transformers.logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)  
    tokenizer =  GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")#bert-base-cased")#(args.pretrained_model)
    tokenizer.pad_token = tokenizer.eos_token
    #model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model = GPTNeoForCausalLM.from_pretrained(args.pretrained_model)
    model.to(device)
    
    sacrebleu = evaluate.load("sacrebleu")
    rouge=evaluate.load('rouge')
    bert=evaluate.load('bertscore')
    if args.pretrain:
        split_train, split_valid, split_test = get_jsn_dialog("../../data/msc/session_2/train.txt", 
                                                        "../../data/msc/session_2/valid.txt",
                                                        "../../data/msc/session_2/test.txt")
        split_train2, split_valid2, split_test2 = get_jsn_dialog("../../data/msc/session_3/train.txt", 
                                                        "../../data/msc/session_3/valid.txt",
                                                        "../../data/msc/session_3/test.txt")
        split_train3, split_valid3, split_test3 = get_jsn_dialog("../../data/msc/session_4/train.txt", 
                                                        "../../data/msc/session_4/valid.txt",
                                                        "../../data/msc/session_4/test.txt")
        split_train4, split_valid4, split_test4 = get_wow_dialog("../../data/wow/train.json", 
                                                        "../../data/wow/valid_random_split.json",
                                                        "../../data/wow/test_random_split.json")

        split_train = split_train+split_train2+split_train3+split_train4
        split_valid = split_valid+split_valid2+split_valid3+split_valid4
        split_test = split_test+split_test2+split_test3+split_test4
    else:
        split_train, split_valid, split_test = get_txt_dialogs(args.train_addr, args.valid_addr, args.test_addr)
    print(len(split_train))
    train_data, valid_data, test_data = data_processing(split_train,split_valid, split_test, tokenizer)
    train_dataset = DailyDialogueDataset(train_data)
    valid_dataset = DailyDialogueDataset(valid_data)
    test_dataset = DailyDialogueDataset(test_data)
    arguments = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy = 'epoch',
        learning_rate=args.lr,
        per_device_train_batch_size= args.batch_size,
        per_device_eval_batch_size=2,
        num_train_epochs= args.epoch,
        weight_decay=args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        seed = args.seed,
        data_seed = args.data_seed,
        load_best_model_at_end = True,
        save_strategy='epoch',
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
    if not args.test:    
        trainer.train()#(resume_from_checkpoint="../model2/checkpoint-6858")
    else:
        pred = trainer.predict(test_dataset)
        print(pred[2])

    