import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
from tqdm import tqdm
from dataset import get_data
import numpy as np
import math


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast = True)
tokenizer.pad_token = tokenizer.eos_token

load_data = get_data( tokenizer = tokenizer )
dataset = load_data()

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=quantization_config

    )
ref_model = AutoModelForCausalLM.from_pretrained(
    'gpt2-medium',
    quantization_config=quantization_config
    )
def compute_score(
        model_input : dict ,
        model_name = None ,
        model = None ,
        for_each_token = False,
        return_normalized = False
    ):

    from transformers import AutoModelForCausalLM

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model is None :
        model = AutoModelForCausalLM.from_pretrained( model_name )

    model = model.to(device)
    model_input = next(iter(model_input))
    with torch.no_grad():
        output = model( **model_input )
        logits = output.logits
    '''
    logits --> ( Batch , SeqLen , Vocab_size )
    log_softmax ----> ( Batch, SeqLen , Vocab_size )
    but diff for (1:4) (prompt:responses)
    '''

    log_softmax = -torch.nn.functional.log_softmax(logits, dim = -1 )
    seq_scores = []
    for input_id in tqdm(range(model_input['input_ids'].shape[2])):
        seq_scores.append( log_softmax[:, :,input_id - 1 , input_id ] )

    if for_each_token :
        return torch.tensor(seq_scores)
    scores = torch.mean(torch.stack(seq_scores, dim = -1),-1).to(torch.float32)
    min = scores.min(dim = -1,keepdim = True).values
    max = scores.max(dim = -1,keepdim = True).values

    if return_normalized:
        scores = (scores - min)/ (max  - min)
    return scores

#print(compute_score(model_input=dataset, model_name='gpt2'))

class rpo_loss_func:
    def __init__( self, ref_model= ref_model, policy_model = model , beta = 0.3, distance_matrix= None ,reward_scaling = 0.3 ):
        self.ref_model = ref_model
        self.policy_model = policy_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.beta = torch.tensor( beta, device = self.device )
        self.D = distance_matrix

        self.reward_scaling = torch.tensor(reward_scaling,device = self.device)

    def backward_kl( self,dataset, reward_k_response):
        '''
        first we are need to compute π(yk∣x) likehood of the model to generate a prompt
        that is done by computing the sum of all the token probability that have in respone
        '''

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


        loss = reward_model_prob * torch.log( reward_model_prob / model_reward_prob )
        return torch.sum( loss)


loss_func = rpo_loss_func()
z = torch.tensor([
        [0.423,0.234,0.2342,0.234],
        [0.234,0.2342,0.2342,0.23412],
        [0.2342,0.634,0.23426,0.634]
    ])

loss_func.backward_kl( dataset , z)
# ''' for backward '''
# loss.backward()
