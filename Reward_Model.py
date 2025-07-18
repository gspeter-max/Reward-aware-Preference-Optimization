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

data = load_dataset('Anthropic/hh-rlhf')['test'] 

batch_size = 300
if tokenized_data.input_ids.shape[0] % 3 != 0: 
    RuntimeError('batch size is not correct') 

data = data.to_pandas().astype(str).values.flatten().tolist()
tokenized_data = tokenizer(data, truncation= True, padding = 'max_length', return_tensors = 'pt').to('cuda')
dataset = TensorDataset(tokenized_data.input_ids,tokenized_data.attention_mask)

data = DataLoader(dataset, batch_size = batch_size)
optim = torch.optim.Adam( reward_model.Parameters() , lr = 0.00234 )

for index, batched_data in enumerate(dataset):
    reward_scores = reward_model(
            input_ids = batched_data[0], 
            attention_mask= batched_data[1]
            ) 
    y1 = reward_scores[0::2]
    y2 = reward_scores[1::2]
    x = y1 - y2
    loss = -torch.nn.functional.log_sigmoid(x) 
    loss.backward()
    optim.step()
    optim.zero_grad()

