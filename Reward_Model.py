from transformers import AutoModelForCausalLM  ,AutoTokenizer 
from torch.utils.data import DataLoader, TensorData
import torch.nn as nn
import torch
from datasets import load_dataset
import numpy as np

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

data = load_dataset('allenai/reward-bench')['filtered']
data = data.remove_columns(['chosen_model','rejected_model','subset','id'])

data = data.to_pandas().astype(str).values.tolist() 
tokenized_data = tokenizer(data, truncation= True, padding = 'max_length', return_tensors = 'pt')

prompt = [] 
chosen = [] 
rejected = [] 

for i in range(int(tokenized_data.input_ids.shape[0]/3)):
    prompt.append((tokenized_data.input_ids[i*3,:],tokenized_data.attention_mask[i*3,:])) 
    chosen.append((tokenized_data.input_ids[i*3+1,:], tokenized_data.attention_mask[i*3+1,:])) 
    rejected.append((tokenized_data.input_ids[i*3+2,:], tokenized_data.attention_mask[i*3+2,:])) 

prompt = np.array(prompt) 
chosen = np.array(chosen) 
rejected = np.array(rejected) 

tensordata = TensorData(prompt,chosen, rejected)

dataset = DataLoader(tensordata , batch_size = 10)
dataset = next(iter(dataset))
result = reward_model(dataset)

optim = torch.optim.Adam( reward_model.Parameters() , lr = 0.00234 )

loss_func = reward_loss( reward_model)
for train_data in torch_dataset:
    prompt , chosen, rejected = train_data 
    loss = loss_func( chosen , rejected )   
    optim.step()
    optim_zero_grad()
    break

