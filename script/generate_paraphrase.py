import argparse
from operator import add
from typing import List
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from transformers import GPT2Tokenizer, GPTNeoForCausalLM, TopKLogitsWarper
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from torch.optim import AdamW

#decoding algorithm
def get_topk_token(logits, embeding, topk, temperature, device):
    if topk == 1:
        return torch.topk(logits, topk, dim=-1).indices[:, 0, :], torch.matmul(F.softmax(logits/0.01, dim=-1).to(device), embeding.weight)
    else:
        topk_indices = logits.topk(k=topk, dim=-1).indices
        mask = torch.zeros_like(logits).scatter_(-1, topk_indices, 1.0)
        masked_logits = logits + mask.log()
        masked_logits = masked_logits/temperature
        uniform_noise = torch.rand_like(masked_logits)
        gumbel_noise = -torch.log(-torch.log(uniform_noise))
        sampled_indices = (masked_logits + gumbel_noise).argmax(dim=-1)
        return sampled_indices, embeding(sampled_indices)

def batch_decode(logits, tokenizer):
    indice = torch.topk(logits, 1, dim=-1).indices
    word = tokenizer.batch_decode(indice[:, :, 0])
    return word
def truncate_text(text, heading):
    if text == "":
        return text
    elif '<|endoftext|>' in text:
        text = text[:text.find('<|endoftext|>')]
    text= text.strip()
    if heading in text:
        return text[:text.find(heading)]
    elif "A:" in text:
        return text[:text.find("A:")]
    elif "B:" in text:
        return text[:text.find("B:")]
    elif text[-1] == heading[0]:
        return text[:-1]
    else:
        return text

# augmentation function
def generate_augmentation(model, tokenizer,device, o1_text, o2_text, resp, max_length, temperature,
                          top_k, num_passes, future_lr, original_lr, future_ratio, original_ratio,
                          future_gradient_iters, original_gradient_iters, verbose, heading):
    tokenized_o1_text = tokenizer.encode(tokenizer.bos_token + o1_text, return_tensors='pt').to(device)
    tokenized_o2_text = tokenizer.encode(o2_text + tokenizer.eos_token, return_tensors='pt').to(device)
    tokenized_resp = tokenizer.encode(resp, return_tensors='pt').to(device)
    
    candidate_list = []
    length = tokenized_resp.shape[1]
    if max_length<length:
        max_length = length
    o1_onehot = torch.zeros(1, tokenized_o1_text.shape[1], tokenizer.vocab_size, dtype=torch.long, requires_grad=False).to(device)
    o1_onehot.scatter_(2, tokenized_o1_text.unsqueeze(-1), 1)
    
    o2_onehot = torch.zeros(1, tokenized_o2_text.shape[1], tokenizer.vocab_size, dtype=torch.long, requires_grad=False).to(device)
    o2_onehot.scatter_(2, tokenized_o2_text.unsqueeze(-1), 1)
      
    # get unperturbed logits
    last = tokenized_o1_text[:, -1:]
    logits = None
    last_embeds = model.get_input_embeddings()(last)
    if tokenized_o1_text.shape[1] > 1:
                past = model(tokenized_o1_text[:, :-1]).past_key_values
    indices_list = None
    for i in range(length):
        # run model forward to obtain unperturbed logits
        tmp = model(past_key_values=past, inputs_embeds=last_embeds)
        unpert_logits = tmp[0][:, -1:, :]
        past =  tmp[1]
        logits = unpert_logits if logits is None else torch.cat((logits, unpert_logits), dim=1)
        indices, last_embeds = get_topk_token(unpert_logits, model.get_input_embeddings(), 1, temperature, device)
        indices_list = indices if indices_list is None else torch.cat((indices_list, indices), dim=-1)
    if verbose:
        print(f"original model generate: {tokenizer.batch_decode(indices_list)}")
        print('---'*50)
    #train the response logit
    for i in range(num_passes):
        future_logits=logits.clone().detach()
        response_logits=logits.clone().detach()
        future_logits.requires_grad=True
        response_logits.requires_grad=True
        future_logits = get_future_gradient(future_logits, model, tokenizer, o1_onehot, o2_onehot, tokenized_o2_text, 
                                            future_lr, future_gradient_iters,  verbose)
        response_logits = get_response_gradient(response_logits,  tokenizer,  tokenized_resp, 
                                            original_lr, original_gradient_iters,  verbose)
        logits = mix_gradient(future_logits, response_logits, model, tokenizer, tokenized_o1_text,
                              length, max_length, future_ratio, original_ratio, temperature, top_k, device, verbose, heading=False)
        #last time
        # get unperturbed logits
        
        last = tokenized_o1_text[:, -1:]
        last_embeds = model.get_input_embeddings()(last)
        if tokenized_o1_text.shape[1] > 1:
                    past = model(tokenized_o1_text[:, :-1]).past_key_values
        indices_list = None
        
        for i in range(max_length):
            # run model forward to obtain unperturbed logits
            tmp = model(past_key_values=past, inputs_embeds=last_embeds)
            unpert_logits = tmp[0][:, -1:, :]
            past =  tmp[1]
            if i < length:
                pert_logits = (1-future_ratio-original_ratio)*unpert_logits+future_ratio*future_logits[:, i:i+1, :]+original_ratio*response_logits[:, i:i+1, :]
            else:
                pert_logits=unpert_logits            
            indices, last_embeds = get_topk_token(pert_logits, model.get_input_embeddings(), top_k,temperature, device)
            indices_list = indices if indices_list is None else torch.cat((indices_list, indices), dim=-1)
        candidate_list.append(truncate_text(tokenizer.batch_decode(indices_list)[0], heading)) 
    if device == "cuda":
        torch.cuda.empty_cache()
    return candidate_list
def get_future_gradient(logits, model, tokenizer, o1_onthot, o2_onehot, tokenized_o2_text, learning_rate, backward_iters, verbose):
    optimizer = AdamW([logits], lr=learning_rate)
   
    o1_embeds = torch.matmul(o1_onthot.float(), model.get_input_embeddings().weight)
    o2_embeds = torch.matmul(o2_onehot.float(), model.get_input_embeddings().weight)
    for i in range(backward_iters):
        #inputs_catoncate = torch.cat((o1_onthot, logits, o2_onehot), dim=1)
        inputs_embeds = torch.cat((o1_embeds, torch.matmul(logits, model.get_input_embeddings().weight), o2_embeds), dim=1)
        all_logits = model(inputs_embeds=inputs_embeds).logits
        predicted_logits = all_logits[:, -o2_onehot.shape[1]-1:-1, :]
        loss  = torch.nn.CrossEntropyLoss()(
            predicted_logits.view(-1, predicted_logits.shape[-1]),
            tokenized_o2_text.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if verbose:
        print(f"future back prop: {batch_decode(logits, tokenizer)}, loss: {float(loss)}")
    return logits
def get_response_gradient(logits,  tokenizer,  tokenized_resp, 
                                    learning_rate, backward_iters, verbose):
    optimizer = AdamW([logits], lr=learning_rate)
    for i in range(backward_iters):
        loss  = torch.nn.CrossEntropyLoss()(
            logits.view(-1, logits.shape[-1]),
            tokenized_resp.view(-1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    if verbose:
        print(f"original back prop: {batch_decode(logits, tokenizer)}, loss: {float(loss)}, {tokenizer.batch_decode(tokenized_resp)}")
    return logits

def mix_gradient(future_logits, response_logits, model, tokenizer,
    tokenized_o1_text, length, max_length, future_ratio, original_ratio, temperature, top_k,
    device, verbose, heading=False):
    
    # get unperturbed logits
    last = tokenized_o1_text[:, -1:]
    logits = None
    last_embeds = model.get_input_embeddings()(last)
    if tokenized_o1_text.shape[1] > 1:
                past = model(tokenized_o1_text[:, :-1]).past_key_values
    indices_list = None
    for i in range(length):
        # run model forward to obtain unperturbed logits
        tmp = model(past_key_values=past, inputs_embeds=last_embeds)
        unpert_logits = tmp[0][:, -1:, :]
        past =  tmp[1]
        
        pert_logits = (1-future_ratio-original_ratio)*unpert_logits+future_ratio*future_logits[:, i:i+1, :]+original_ratio*response_logits[:, i:i+1, :]
        #pert_logits = original_ratio*response_logits[:, i:i+1, :]+0*future_logits[:, i:i+1, :]+0*unpert_logits
        
        logits = pert_logits if logits is None else torch.cat((logits, pert_logits), dim=1)
        indices, last_embeds = get_topk_token(pert_logits, model.get_input_embeddings(), 1,temperature, device)
        indices_list = indices if indices_list is None else torch.cat((indices_list, indices), dim=-1)
    if verbose:
        print(tokenizer.batch_decode(indices_list))
        print('---'*50)
    return logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # general tunable parameters
    parser.add_argument(
        "--pretrained_model", type=str, default="../train_model/model/bestGPT2",
        help="Pretrained model name or path to local checkpoint")
    parser.add_argument("--max_length", type=int, default=40, help="Max length of generated text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_k", type=int, default=1, help="Top-k sampling.")
    parser.add_argument("--unable_version_2", action="store_true", help="whether not to use character-sentence pattern")
    # tunable parameters for different gradient methods
    parser.add_argument("--future_ratio", type=float, default=0.12, help="Proportion of the gradient from future dialogue")
    parser.add_argument("--original_ratio", type=float, default=0.12, help="Proportion of the gradient from original response")
    parser.add_argument("--future_lr", type=float, default=0.0004, help="Learning rate for future response gradient.")
    parser.add_argument("--original_lr", type=float, default=0.02, help="Learning rate for the original response gradient")
    parser.add_argument("--future_gradient_iters", type=int, default=20, help="Number of iterations updating future gradient")
    parser.add_argument("--original_gradient_iters", type=int, default=12, help="Number of iterations updating original gradient")
    parser.add_argument("--num_passes", type=int, default=20, help="Number of passes to both get the gradients and merging gradients.")
    # untunable parameters
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--verbose", action="store_true", help="Print intermediate states to help with tuning / debugging.")
    parser.add_argument("--input_file", type=str, default="./script_data/testData_withHint.json", help="Input data in json format.")
    parser.add_argument("--output_file", type=str, default="./script_data/origin.json", help="Output dir.")

    args = parser.parse_args()
    # setup the deivice
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    # seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # load pretrained gpt2 model
    #model = GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model = GPTNeoForCausalLM.from_pretrained(args.pretrained_model)
    model.to(device)
    model.eval()
    # load gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False
    # read input
    input_file = []
    with open(args.input_file, 'r') as read_file:
        read_lines = read_file.readlines()
        input_file = [json.loads(line.strip()) for line in read_lines]
        read_file.close()
    with open(args.output_file, 'w') as output_file:
        for input in tqdm(input_file):
            if args.unable_version_2:
                heading = ''
            else:
                if input['prev'][-2] == 'A':
                    heading = 'B:'
                else:
                    heading = 'A:'
            candidate_list = generate_augmentation(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    o1_text=input['prev'],
                    o2_text=input['future'],
                    resp=input['cur'],
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,                    
                    num_passes=args.num_passes,
                    future_lr=args.future_lr,
                    original_lr=args.original_lr,
                    future_ratio=args.future_ratio,
                    original_ratio=args.original_ratio,
                    future_gradient_iters=args.future_gradient_iters,
                    original_gradient_iters=args.original_gradient_iters,
                    verbose=args.verbose,
                    heading=heading,)
            d = {
                'O1': input['prev'],
                'O2': input['future'],
                'resp': input['cur'],
                'H_Candidates': candidate_list
            }
            output_file.write(json.dumps(d) + '\n')
            output_file.flush()
                