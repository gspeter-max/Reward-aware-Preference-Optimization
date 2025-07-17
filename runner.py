import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers.utils.quantization_config import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast = True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "gpt2",
    quantization_config=quantization_config

    )

# dataset = load_dataset('Amod/mental_health_counseling_conversations')
# def map_tokenizer(example):
#     return tokenizer(
#         text = example['Context'],
#         text_target = example['Response'],
#         padding = 'max_length',
#         truncation = True,
#         return_tensors = 'pt'
#     ).to('cuda')
# dataset = dataset['train'].map(map_tokenizer, batched=True, remove_columns = dataset['train'].column_names)
# dataset.set_format(
#     type = 'torch',
#     columns = ['input_ids','attention_mask', 'labels'],
#     output_all_columns = True
# )

def compute_score( seq ,model_name = None , model = None ,tokenizer = None, for_each_token = False):
     
    if torch.cuda.is_available():
        device= 'cuda' 
    else: 
        device = 'cpu' 

    if model is None :
        model = AutoModelForCausalLM.from_pretrained( model_name ).to(device) 

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained( model_name ) 
    
    model_input = tokenizer( seq , return_tensors = 'pt' ).to(device) 
    with torch.no_grad():
        output = model( **model_input )  
        logits = output.logits
    '''
    logits --> ( Batch , SeqLen , Vocab_size ) 
    log_softmax ----> ( Batch, SeqLen , Vocab_size ) 

    '''

    log_softmax = torch.nn.functional.log_softmax(logits, dim = -1 ) 
    
    seq_scores = []
    for input_id in range(len(model_input['input_ids'])):
        seq_scores.append( log_softmax[:, input_id - 1 , input_id f] )

    seq_scores = torch.stack(seq_scores, dim = 0)
    if for_each_token :
        return seq_scores 
    
    return torch.sum(seq_scores) 


get_model_prob = Generate_Response_Prob( tokenizer )
class rpo_loss_func:
    def __init__( self, ref_model= model, policy_model = model , beta = 0.3, distance_matrix= None ,reward_scaling = 0.3 ):
        self.ref_model = ref_model
        self.policy_model = policy_model
        self.beta = beta
        self.D = distance_matrix
        self.reward_scaling = torch.tensor(reward_scaling)

    def backward_kl( self,y_k,x, reward_k_response):
        '''
        first we are need to compute π(yk∣x) likehood of the model to generate a prompt
        that is done by computing the sum of all the token probability that have in respone
        '''

        raw_reward_prob = self.reward_scaling * reward_k_response

            

        raw_models_prob = self.beta * torch.log( raw_policy_likelihood / raw_ref_model_likelihood )
        model_reward_prob = torch.nn.functional.softmax( raw_models_prob, dim = -1 )
        reward_model_prob = torch.nn.functional.softmax( raw_reward_prob, dim = -1 )

        loss = reward_model_prob * torch.log( reward_model_prob / model_reward_prob )
        return torch.sum( loss, dim = -1)


loss_func = rpo_loss_func()

x = ['hi i am genius i am extremly creative , and you ?']
y = ['yeah i am ', 'i think yes']
z = torch.tensor([0.234,0.43])

loss_func.backward_kl( y, x, z)
''' for backward '''
# loss.backward()
