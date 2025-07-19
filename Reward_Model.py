from transformers import AutoModelForCausalLM  ,AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
from datasets import load_dataset
import numpy as np

pre_model = AutoModelForCausalLM.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


class Reward_Model(nn.Module):
    def __init__(self, model: nn.Module , model_output_shape ):
        super().__init__()
        if model.device != 'cuda':
            if torch.cuda.is_available():
                model = model.to('cuda')

        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.eval()
        self.final_layer = nn.Linear( model_output_shape, 1 ).cuda()
        self.hidden = None

    def forward(self, x, for_eval = False):

        if for_eval:
            hidden_state = self.model(
                input_ids= x['input_ids'],
                attention_mask = x['attention_mask']
            ).logits[:, :,-1,:]
        else:
            hidden_state = self.model(
                input_ids= x[0],
                attention_mask = x[1]
            ).logits[:, -1,:]

        self.hidden = hidden_state
        return self.final_layer(hidden_state )

''' for training reward model ''' 
print(f'reward model trianing start') 
reward_model = Reward_Model( model = pre_model, model_output_shape = pre_model .config.vocab_size ).to('cuda')
data = load_dataset('Anthropic/hh-rlhf')['test']


data = data.to_pandas().astype(str).values.flatten().tolist()
tokenized_data = tokenizer(data, truncation= True, padding = 'max_length', return_tensors = 'pt').to('cuda')
dataset = TensorDataset(tokenized_data.input_ids,tokenized_data.attention_mask)


batch_size = 8

if (tokenized_data.input_ids.shape[0] % batch_size) != 0:
    raise ValueError('batch size is not correct')

data = DataLoader(dataset, batch_size = batch_size)
optim = torch.optim.Adam( reward_model.parameters() , lr = 0.00234 )

for index, batched_data in enumerate(data):
    if index == 8:
        break
    reward_scores = reward_model(batched_data)
    y1 = reward_scores[0::2]
    y2 = reward_scores[1::2]
    x = y1 - y2
    loss = -torch.nn.functional.logsigmoid(x)
    loss = loss.mean()
    loss.backward()
    optim.step()
    optim.zero_grad()
    print(f'loss is : {loss}')
