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

class Generate_Response_Prob:
    def __init__(
            self,
            tokenizer,
            num_responses_for_one_prompt= 2,
            batched_response = 1
            ):
        self.tokenizer = tokenizer
        self.num_responses = num_responses_for_one_prompt
        self.batched_response = batched_response


    def compute_prob( self,model, prompt, generated_response_k ):
        log_prob_list = []
        log_prob = 0

        prompt_tokenized = self.tokenizer(
                prompt,
                return_tensors = 'pt',
                truncation = True ,
                padding = 'max_length'
                ).to('cuda')

        for i in range( self.num_responses ):

            attention_tracking = torch.tensor([
                    (i,self.tokenizer(prompt[i], return_tensors = 'pt').input_ids.shape[-1]) 
                    for i in range(self.batched_response)
                    ])

            generated_response = generated_response_k[i : i+1 ]
            response_tokenized = self.tokenizer(
                    generated_response,
                    return_tensors = 'pt',
                    truncation = True,
                    padding = 'max_length'
                            ).to('cuda')


            response_tokenized = response_tokenized.input_ids

            # if len(response_tokenized.shape) < 3:
            #     print(len(response_tokenized.shape))
            #     response_tokenized = response_tokenized.unsqueeze(0)


            for j in range( self.tokenizer.max_len_single_sentence):

                #if len(prompt_tokenized.input_ids.shape) < 3:
                    #                   print(len(prompt_tokenized.input_ids.shape))
#                    prompt_tokenized.input_ids = prompt_tokenized.input_ids.unsqueeze(0)
                print(j)
                att_trac_row, att_trac_col = zip(*attention_tracking)
                prompt_tokenized.attention_mask[att_trac_row, att_trac_col] = torch.tensor(1) # here the main logic
                model_output = model( **prompt_tokenized ).logits


                soft_max_model_output = torch.nn.functional.softmax(model_output[:,-1,:], dim = -1)
                log_prob = log_prob + torch.log(soft_max_model_output[ :,response_tokenized[:, j ]] )

                prompt_tokenized.input_ids[att_trac_row ,att_trac_col] = response_tokenized[:,j]

                attention_tracking = attention_tracking + 1

                att_trac_row , att_trac_col = zip( *attention_tracking )
                att_trac_row = torch.tensor(att_trac_row ) - 1 
                attention_tracking = torch.tensor(
                        list(
                            zip(
                                att_trac_row, torch.tensor(att_trac_col) 
                                )
                            ) 
                        )
                                                
            print(f'thing one ')
            log_prob_list.append( log_prob )

        return torch.tensor(
                log_prob_list,
                requires_grad = True
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
