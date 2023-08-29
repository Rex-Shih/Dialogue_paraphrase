import numpy as np
import json
import random
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

def data_processing(split_train, split_valid, split_test):
    cand= {'A':[], 'B':[]}
    def proc_txt(file):
        ret = []
        allsent = []     
        for sent in file: 
            for turn in range(2):
                #decide which is prev, which is future
                rand_num = np.random.randint(1, len(sent), size=1)[0]
                first = ""
                second=""
                second_split = []
                first_split = []
                for i, content in enumerate(sent):
                    if i %2 == 0:
                        tmp = "A: "
                    else:
                        tmp = "B: "
                    
                    if i<rand_num:
                        #prev
                        first = first+" "+tmp+content
                        first_split.append(content)
                    else:
                        #future
                        if i == rand_num:
                            second_heading = tmp[0]
                        second = second+" "+tmp+content
                        second_split.append(content)
                cand[second_heading].append(second[1:])
                ret.append([first[1:], second[1:], second_heading, second_split, first_split])
                allsent.append(first[1:])
                allsent.append(second[1:])        
        return ret, allsent
    def split_dialogue(file_dataset, allsent):
        ret = []
        for dialogue in file_dataset:
            tmp = {}
            tmp['sent_a']=dialogue[0]
            tmp['sent_b']=dialogue[1]
            tmp['label']=0
            ret.append(tmp)
            tmp = {}
            tmp['sent_a']=dialogue[0]
            num = np.random.randint(0, len(allsent), size=1)[0]
            if num %4 == 3:# or dialogue[2] == 100:
                while(allsent[num]==dialogue[1]):# or allsent[num] == dialogue[0]):
                    num = np.random.randint(0, len(allsent), size=1)[0]
                tmp['sent_b']=dialogue[2]+allsent[num][1:]
            elif num%4 == 2:
                sent = ""
                if dialogue[2] == 'A':
                    snd = 'B'
                else:
                    snd = 'A'
                for i, first_split in enumerate(dialogue[4]):
                    if i %2 == 0:
                        heading = f"{dialogue[2]}: "
                    else:
                        heading = f"{snd}: "
                    sent = sent+' '+heading+dialogue[4][i]
                
                tmp['sent_b']=sent[1:]
            else:
                split = num = np.random.randint(0, len(dialogue[3]), size=1)[0]
                if dialogue[2] == 'A':
                    snd = 'B'
                else:
                    snd = 'A'
                i = 0
                sent = ""
                rand = np.random.randint(0, 2, size=1)[0]
                while(True):
                    if i %2 == 0:
                        heading = f"{dialogue[2]}: "
                    else:
                        heading = f"{snd}: "
                    if rand == 1:
                        if i <=split:
                            sent = sent+' '+heading+dialogue[3][i]
                        else:
                            num = np.random.randint(0, len(cand[heading[0]]), size=1)[0]
                            sent = sent[1:]+' '+cand[heading[0]][num]
                            break
                    else:
                        if i == len(dialogue[3]):
                            sent = sent[1:]
                            break
                        if i == split:
                            sent = sent+' '+dialogue[3][i]
                            tmp_switch = dialogue[2]
                            dialogue[2] = snd
                            snd = tmp_switch
                        else:
                            sent = sent+' '+heading+dialogue[3][i]
                    i = i+1
                tmp['sent_b'] = sent
            tmp['label']=1    
            ret.append(tmp)
        return ret
    train_dialogue, train_allsent = proc_txt(split_train)
    valid_dialogue, valid_allsent = proc_txt(split_valid)
    test_dialogue, test_allsent = proc_txt(split_test)
    Tok_TrainData = split_dialogue(train_dialogue, train_allsent)
    Tok_ValidData = split_dialogue(valid_dialogue, valid_allsent)
    Tok_TestData = split_dialogue(test_dialogue, test_allsent)
    return Tok_TrainData, Tok_ValidData, Tok_TestData

np.random.seed(0)
random.seed(0)

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


train_data, valid_data, test_data = data_processing(split_train, split_valid, split_test)
 
print(len(split_train))
print(len(valid_data))
print(len(test_data))
"""with open("./data/pretrain_NSP.json", 'w') as fw:
    for i in train_data:
        fw.write(json.dumps(i) + '\n')
    fw.close()
with open("./data/prevalid_NSP.json", 'w') as fw:
    for i in valid_data:
        fw.write(json.dumps(i) + '\n')
    fw.close()
with open("./data/pretest_NSP.json", 'w') as fw:
    for i in test_data:
        fw.write(json.dumps(i) + '\n')
    fw.close()"""
    