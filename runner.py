import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from tqdm import tqdm
from typing import Union
import torch.nn as nn
from Reward_Model import Reward_Model
from dataset import get_data
import numpy as np
import math


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_quant_type = 'nf4', 
    bnb_4bit_compute_dtype= torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast = True)
tokenizer.pad_token = tokenizer.eos_token

load_data = get_data( tokenizer = tokenizer )
dataset = load_data()

pre_model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=quantization_config
    )

# ref_model = AutoModelForCausalLM.from_pretrained(
#     'gpt2-medium'
#     # quantization_config=quantization_config
#     )
def compute_score(
        model_input : dict ,
        model_name = None ,
        model = None ,
        for_each_token = False,
        return_normalized = False
    ):

    from transformers import AutoModelForCausalLM

    if model is None :
        model = AutoModelForCausalLM.from_pretrained( model_name )
    else:
        if model.device != 'cuda':
            if torch.cuda.is_available():
                model = model.to('cuda') 

    model_input = next(iter(model_input))
    with torch.no_grad():
        output = model( **model_input )
        logits = output.logits
    '''
    logits --> ( Batch , SeqLen , Vocab_size )
    log_softmax ----> ( Batch, SeqLen , Vocab_size )
    but diff for (1:2) (prompt:responses)
    '''

    log_softmax = -torch.nn.functional.log_softmax(logits, dim = -1 )
    seq_scores = []
    for input_id in tqdm(range(model_input['input_ids'].shape[2])):
        seq_scores.append( log_softmax[:, :,input_id - 1 , input_id ] )

    if for_each_token :
        return torch.tensor(seq_scores)
    scores = torch.mean(torch.stack(seq_scores, dim = -1),-1).to(torch.float32)

    if return_normalized:
        min = scores.min(dim = -1,keepdim = True).values
        max = scores.max(dim = -1,keepdim = True).values

        scores = (scores - min)/ (max  - min)
    return scores

#print(compute_score(model_input=dataset, model_name='gpt2'))

class rpo_loss_func:
    def __init__( 
            self, 
            ref_model : nn.Module = pre_model,
            policy_model : nn.Module = pre_model ,
            beta : Union[int,float ] = 0.3,
            distance_matrix= None ,
            reward_scaling = 0.3 
        ):
        
        self.ref_model = ref_model
        self.policy_model = policy_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.beta = beta
        self.D = distance_matrix
        self.reward_scaling = reward_scaling

    def backward_kl( self,dataset, reward_k_response):

        if self.device != 'cuda':
            if torch.cuda.is_available():
                if self.ref_model.device != 'cuda':
                    self.ref_model = self.ref_model.to('cuda')
                if self.policy_model.device != 'cuda':
                    self.policy_model = self.policy_model.to('cuda') 
                if not isinstance(self.beta, torch.Tensor): 
                    self.beta = torch.tensor(self.beta, device = 'cuda')
                if not isinstance(self.reward_scaling, torch.Tensor):
                    self.reward_scaling = torch.tensor(self.reward_scaling, device = 'cuda')
                if not isinstance( reward_k_response, torch.Tensor):
                    reward_k_response = reward_k_response.to(self.device)

        raw_reward_prob = self.reward_scaling * reward_k_response

        raw_policy_likelihood = compute_score(
                model_input = dataset,
                model = self.policy_model
                )
        raw_ref_model_likelihood = compute_score(
                model_input = dataset,
                model = self.ref_model
                )

        raw_models_prob = self.beta * torch.log( raw_policy_likelihood / raw_ref_model_likelihood )
        model_reward_prob = torch.nn.functional.softmax( raw_models_prob, dim = -1 )
        reward_model_prob = torch.nn.functional.softmax( raw_reward_prob, dim = -1 )

        # here we are doing (backward (reverse ) kl divergence)  
        loss = reward_model_prob * torch.log( reward_model_prob / model_reward_prob )
        return torch.sum( loss)


reward_model = Reward_Model( model = pre_model, model_output_shape = pre_model.config.vocab_size ).to('cuda')
for param in reward_model.named_parameters():
    if param[1].dtype != torch.bfloat16:
        param[1].data = param[1].data.to(torch.bfloat16)

iter_dataset = iter(dataset)
next_dataset = next(iter_dataset)
# print(next_dataset['input_ids'].dtype)
# print(next_dataset['attention_mask'].dtype)
result  = reward_model(next_dataset, for_eval  = True)
B,S,V = result.shape
reward_score = result.reshape(B,S*V) 

loss_func = rpo_loss_func()
loss = loss_func.backward_kl( dataset , reward_score)
print(f'loss is : -- {loss}') 
# ''' for backward '''
# loss.backward()



