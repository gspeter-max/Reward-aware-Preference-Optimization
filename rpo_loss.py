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
        #if not isinstance( reward_k_response, torch.Tensor ):
            #            reward_k_response = torch.tensor(reward_k_response)

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
#        raw_models_prob = torch.tensor([ self.beta * torch.log( rpm / rfm ) 
                                        #                for rpm , rfm in zip( raw_policy_likelihood , raw_ref_model_likelihood ) 
                                        #       ]) 
        raw_models_prob = self.beta * torch.log( raw_policy_likelihood / rew_ref_model_likelihood )
        model_reward_prob = torch.nn.functional.softmax( raw_models_prob, dim = -1 )
        reward_model_prob = torch.nn.functional.softmax( raw_reward_prob, dim = -1 )
        
#        loss = torch.tensor([rmp * torch.log( rmp / mrp ) for rmp, mrp in zip( reward_model_prob, model_reward_prob)]) 
        loss = reward_model_prob * torch.log( reward_model_prob/ model_reward_prob ) 
        return torch.sum( loss, dim = -1)

loss_func = rpo_loss_func()

x = 'hi i am genius i am extremly creative? '
y = ['yeah i am ', 'i think yes']
z = torch.tensor([0.234,0.43]) 

loss_func.backward_kl( y, x, z)

    # def __call__(self ,
# from GL_probility import G

