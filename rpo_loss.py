import torch 
from GL_probility import G
class rpo_loss_func:
        def __init__( self, ref_model , policy_model , beta, distance_matrix ):
        self.ref_model = ref_model 
        self.policy_model = policy_model 
        self.beta = beta 
        self.D = distance_matrix 
    
    def backward_kl( self,y,x):
        '''
        first we are need to compute π(yk∣x) likehood of the model to generate a prompt 
        that is done by computing the sum of all the token probability that have in respone 
        '''

        model_respone_likelihood = 
        exp_response_term = torch.exp( self.beta * torch.log(

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
