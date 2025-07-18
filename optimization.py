from transformers import AutoModelForCausalLM  ,AutoTokenizer 
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from datasets import load_dataset


model = AutoModelForCausalLM.from_pretrained(
       'gpt2' 
        )
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token 


class Reward_Model(nn.Module): 
    def __init__(self, model: nn.Module , model_output_shape = model.config.vocab_size ):
        super().__init__() 
        self.model = model 
        self.final_layer = nn.Linear( model_output_shape, 1 )
        self.hidden = None 

    def forward(self, x):
        hidden_state = self.model(**x).logits[:, -1,:]
        self.hidden = hidden_state 
        return self.final_layer(hidden_state ) 

class reward_loss:
    def __init_( self, model):
        self.model = model 

    def __call__( self, y2, y1):
        y2_reward = model(y2) 
        y1_reward = model(y1) 

        x = y1 - y2 
        return torch.nn.functional.log_sigmoid(x)

reward_model = Reward_Model( model = model, model_output_shape = model.config.vocab_size ).to('cuda')
# tokenized = tokenizer(
#          text = ['hi i am a genius', 'oki doki '] , 
#          padding = 'max_length',
#          truncation = True,
#          return_tensors = 'pt'
#      ).to('cuda')
# result = reward_model(tokenized)
# print(reward_model.hidden) 


dataset = load_dataset('Amod/mental_health_counseling_conversations',split='train')
def map_function(example):
    combine_features = [f'{context} [SEP] {response}' for context, response in zip(example['Context'], example['Response'])] 
    tokenized = tokenizer(
        text = combine_features, 
        truncation = True, 
        padding = 'max_length'
    )
    return tokenized 

dataset  = dataset.map( map_function , batched = True , remove_columns= dataset.column_names)
dataset.set_format(
    type = 'torch', 
    device = 'cuda'
)

dataset = DataLoader(dataset, batch_size = 10)
dataset = next(iter(dataset))
result = reward_model(dataset)

optim = torch.optim.Adam( reward_model.Parameters() , lr = 0.00234 )
''' you must be pass models over here '''

loss_func = reward_loss( reward_model) 
for train_data in torch_dataset:
    loss = loss_func( y1_data , y2_data ) 
    optim.step() 
    optim_zero_grad()
    break

