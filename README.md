# Dialogue Paraphrase

Reference Paper: [Back to the Future: Unsupervised Backprop-based Decoding for Counterfactual and Abductive Commonsense Reasoning](https://arxiv.org/pdf/2010.05906.pdf)

### Run the code
please see [run_code.ipynb](https://github.com/Rex-Shih/Dialogue_paraphrase/blob/main/run_code.ipynb). The file contains a detailed guide written in Jupyter Notebook format.

### Task Description

The goal is to paraphrase a single utterance from a speaker in a two-person dialogue. What sets this apart from a standard paraphrasing task is the need for consistency and fluency to connect **previous and future dialogue.**  
For example:(from ENMLP dataset)  
<img src="https://github.com/Rex-Shih/Dialogue_paraphrase/blob/main/assets/paraphrase_example.png" alt="paraphrase example" width="500"/>

In this example, merely paraphrasing the sentence without considering the **previous/future dialogue** would result in the loss of critical information, which is the ongoing taxi drivers' strike.

### Our Method
Based on the approach outlined in the referenced paper,, we combine three different logits to produce the final paraphrase through the following steps:  
1. We use GPT2, employing the **previous dialogue** as a prompt to generate <span style="color:blue">logits-1</span> which contain the information of the **previous dialogue**. These logits have a fixed length that matches the length of the original paraphrasing sentence.
2. We then continue generating <span style="color:red">logits-2</span> until their length equals that of the **future dialogues**.
3. Next, we employ cross-entropy to compare <span style="color:red">logits-2</span> with the future logits (tokenized **future dialogue**), and use **backpropagation** to fine-tune <span style="color:blue">logits-1</span>. By doing so, we generate logits that capture both the previous and future dialogue information.  
<img src="https://github.com/Rex-Shih/Dialogue_paraphrase/blob/main/assets/model_diagram.png" alt="model Diagram" width=600/>  


5. Using the fine-tuned <span style="color:blue">logits-1</span>, we undertake a cross-entropy comparison with the original paraphrase sentence, followed by another fine-tuning on <span style="color:blue">logits-1</span>.
6. The process from step 1 to step 4 is called one round of update. In our experience, we conduct 2 update rounds in total. Following each update, we sample one sentence from fine-tuned <span style="color:blue">logits-1</span>. The sampled sentence is called **candidate sentence**.
7. Finally, we use the pretrained NSP model (BERT-based-cased) to select the representative sentence from the candidate sentences, by calculate the NSP score:  
the representative sentence serves as the final paraphrased sentence.
<img src="https://github.com/Rex-Shih/Dialogue_paraphrase/blob/main/assets/paraphrae_generate.png" width=300>  

### Evaluation

We use many conversational datasets to train a NSP model, and get the score from past utterance to current paraphrase utterance, as well as the score of current paraphrase utterance to future utterance.
