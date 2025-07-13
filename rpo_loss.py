import torch 
from GL_probility import G
class rpo_loss_func:
    def __init__( self, ref_model , policy_model , beta, distance_matrix ,reward_scaling ):
        self.ref_model = ref_model 
        self.policy_model = policy_model 
        self.beta = beta 
        self.D = distance_matrix 
        self.reward_scaling = reward_scaling 
    
    def backward_kl( self,y_k,x, reward_k_response: list ):
        '''
        first we are need to compute π(yk∣x) likehood of the model to generate a prompt 
        that is done by computing the sum of all the token probability that have in respone 
        '''

        raw_reward_prob = self.reward_scaling * reward_k_response
        raw_policy_likelihood = get_model_prob.compute_prob(
                self.policy_model, 
                x, 
                y_k
            )
        raw_ref_model_likelihood = get_model_prob.compute_prob(
                self.ref_model, 
                x, 
                y_k 
            ) 
        raw_models_prob = [ self.beta * torch.log( rpm / rfm ) 
                for rpm , rfm in zip( raw_policy_likelihood , raw_ref_model_likelihood ) 
                ]
        model_reward_prob = torch.nn.functional.softmax( raw_models_prob, dim = -1 ) 
        reward_model_prob = torch.nn.functional.softmax( raw_reward_prob, dim = -1 )
        
        loss = [rmp * torch.log( rmp / mrp ) for rmp, mrp in zip( reward_model_prob, model_reward_prob)]
        return torch.sum( loss, dim = -1)

    def __call__(self , 



from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", use_fast = True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/DialoGPT-medium",
    quantization_config=quantization_config

    )

dataset = load_dataset('Amod/mental_health_counseling_conversations')
def map_tokenizer(example):
    return tokenizer(
        text = example['Context'],
        text_target = example['Response'],
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt'
    ).to('cuda')
dataset = dataset['train'].map(map_tokenizer, batched=True, remove_columns = dataset['train'].column_names)
dataset.set_format(
    type = 'torch', 
    columns = ['input_ids','attention_mask', 'labels'], 
    output_all_columns = True 
)
