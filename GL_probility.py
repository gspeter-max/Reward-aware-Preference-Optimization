import torch

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
        attention_tracking = torch.tensor(
                [
                    self.tokenizer(prompt[i]).input_ids.shape[-1] for i in range(len(prompt))
                    ]    
                )

        prompt_tokenized = self.tokenizer(
                prompt, 
                return_tensors = 'pt', 
                truncation = True , 
                padding = 'max_length'
                ).to('cuda') 

        for i in range( self.num_responses ):
            generated_response = generated_response_k[i : i+1 ]
            response_tokenized = self.tokenizer(
                    generated_response,
                    return_tensors = 'pt',
                    truncation = True,
                    padding = 'max_length'
                            ).to('cuda')

            
            response_tokenized = response_tokenized.input_ids

            if len(response_tokenized.shape) < 3: 
                print(len(response_tokenized.shape))
                response_tokenized = response_tokenized.unsqueeze(0)

            print(f'response_tokenized : {response_tokenized.shape}')

            for j in range( self.tokenizer.max_len_single_sentence):
                print(f'that is for J : {j}')

                if len(prompt_tokenized.input_ids.shape) < 3: 
                    print(len(prompt_tokenized.input_ids.shape))
                    prompt_tokenized.input_ids = prompt_tokenized.input_ids.unsqueeze(0)

                print(f'prompt_tokenized : {prompt_tokenized.input_ids.shape}')
                prompt_tokenized.attention_mask[attention_tracking] = 1 # here the main logic 

                model_output = model( **prompt_tokenized ).logits
                print(f'model output : {model_output.shape}')
                soft_max_model_output = torch.nn.functional.softmax(model_output[:,-1,:], dim = -1)
                print(f'soft_max_model_output :" {soft_max_model_output.shape}')
                log_prob = log_prob + torch.log(soft_max_model_output[ :,response_tokenized[:,:, j ] ] )
                print(f'log prob :" {log_prob}')
                prompt_tokenized.input_ids = torch.cat(
                        prompt_tokenized.input_ids, 
                        response_tokenized[:,:,j], 
                        dim = 0 
                        ) 

                attention_tracking = attention_tracking + 1 
            log_prob_list.append( log_prob )

        return torch.tensor(
                log_prob_list,
                requires_grad = True
                )

