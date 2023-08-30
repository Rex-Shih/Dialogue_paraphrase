import json
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=314)
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)

dic = []
data_withHint = []
with open('../data/EMNLP_dataset/test/dialogues_test.txt', 'r') as f:
  for i in f:
    
    txt_obj = i.split(' __eou__ ')
    txt_obj[-1] = txt_obj[-1].split(' __eou__\n')[0]
    if len(txt_obj)<=2:
      continue
        
    for i , sent in enumerate(txt_obj):
      if i%2 == 0:
        heading = "A: "
      else:
        heading = "B: "
        
      txt_obj[i] = heading+sent+' '
    
    
    ran_int = np.random.randint(len(txt_obj)-2, size=1)[0]+1
    prev = "".join(txt_obj[0:ran_int])
    cur=txt_obj[ran_int]
    future="".join(txt_obj[ran_int+1:])
    dic.append({
      "prev":prev,
      "cur":cur,
      "future":future
    })
    data_withHint.append({
      "prev":prev+cur[:2],
      "cur":cur[2:],
      "future":future[:-1]
    })
    
f.close()
with open('./script_data/testData.json', 'w') as f:
  for i in dic:
    f.write(json.dumps(i) + '\n')
f.close()
with open('./script_data/testData_withHint.json', 'w') as f:
  for i in data_withHint:
    f.write(json.dumps(i) + '\n')
f.close()