import argparse
from operator import add
from transformers import BertTokenizer, BertForMaskedLM, TrainingArguments, Trainer, BertForNextSentencePrediction, AdamW, RobertaForSequenceClassification, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import csv
from tqdm import tqdm
import random
import transformers
from transformers import RobertaTokenizer, XLNetTokenizer
import torch.nn.functional as F

def data_processing(train_addr, valid_addr, test_addr, tokenizer):
    def split_dialogue(file_dataset):
        sent_a = []
        sent_b = []
        labels = []
        for dialogue in file_dataset:
            dialogue = json.loads(dialogue)
            sent_a.append(dialogue['sent_a'])
            sent_b.append(dialogue['sent_b'])
            labels.append(dialogue['label'])
        ret =  tokenizer(sent_a, sent_b, return_tensors='pt', max_length=512, truncation="longest_first", padding='max_length')
        ret['label'] = torch.LongTensor([labels]).T
        return ret
    train = open(train_addr, 'r')
    valid = open(valid_addr, 'r')
    test = open(test_addr, 'r')
    Tok_TrainData = split_dialogue(train)
    Tok_ValidData = split_dialogue(valid)
    Tok_TestData = split_dialogue(test)
    
    return Tok_TrainData, Tok_ValidData, Tok_TestData

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}
        
class DailyDialogueDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model", type=str, default="bert-base-cased",
        help="pretrained model name or path to local checkpoint")
    parser.add_argument("--epoch", type=int, default=10,help="Training epoch.")
    parser.add_argument("--batch_size", type=int, default=8,help="Training batch size")
    parser.add_argument("--train_addr", type=str, default="./data/train_NSP.json", help="train dataset")
    parser.add_argument("--valid_addr", type=str, default="./data/valid_NSP.json", help="evaluation dataset")
    parser.add_argument("--test_addr", type=str, default="./data/test_NSP.json", help="test dataset")
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
    
    args = parser.parse_args()
    #transformers.logging.set_verbosity_error()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)  
    tokenizer =  AutoTokenizer.from_pretrained("roberta-large")#bert-base-cased")#(args.pretrained_model)
    model = RobertaForSequenceClassification.from_pretrained(args.pretrained_model)
    model.to(device)
    
    train_data, valid_data, test_data = data_processing(args.train_addr, args.valid_addr, args.test_addr, tokenizer)
    train_dataset = DailyDialogueDataset(train_data)
    valid_dataset = DailyDialogueDataset(valid_data)
    test_dataset = DailyDialogueDataset(test_data)
    
    arguments = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy = 'epoch',
        learning_rate=args.lr,
        per_device_train_batch_size= args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs= args.epoch,
        weight_decay=args.weight_decay,
        lr_scheduler_type = args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        seed = args.seed,
        data_seed = args.data_seed,
        load_best_model_at_end = True,
        metric_for_best_model = "accuracy" ,
        save_strategy='epoch',
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        eval_accumulation_steps=2
        )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        )
    if not args.test:    
        trainer.train()#resume_from_checkpoint="../model2/checkpoint-82296")
    else:
        pred = trainer.predict(test_dataset)
        print(pred[2])
        pred = trainer.predict(valid_dataset)
        print(pred[2])
        """ acc = 0
        for i, prediction in enumerate(pred[0]):
            next_sent_pred = np.argmax(prediction, axis=-1)
            if test_dataset[i]['label'] == next_sent_pred:
                acc+=1
        print(acc/len(test_dataset))"""
