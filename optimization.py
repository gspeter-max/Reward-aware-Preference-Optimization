import torch 
from torch.utils.data import DataLoader
import torch.nn as nn 

model = AutoModelForCasualLM.from_pretrained(
       'gpt2' 
        )

class Reward_Model(nn.Module): 
    def __init__(self, model: nn.Module , model_output_shape ):
        self.model = model 
        self.final_layer = torch.Linear( model_output_shape, 1 , requires_grad = True )

    def forward(self, x):
        hidden_state = self.model(x).logits[:, -1,:] 
        return self.final_layer(hidden_state ) 

reward_model = Reward_Model( model = model, model_output_shape = model.config.vocab_size )
'''
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
'''
torch_dataset = DataLoader(
        dataset, 
        batch_size = 40
        ) 
optim = torch.optim.Adam( model.Parameters() , lr = 0.00234 )
''' you must be pass models over here '''

loss_func = rpo_loss_func()  
for train_data in torch_dataset:
    loss = loss_func.backward_kl( 
                train_data['Context'],
                train_data['Response'],
                reward_model(train_data['Response'] ) 
                )
    loss.backward() 
    optim.step() 
    optim_zero_grad() 


                
