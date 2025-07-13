import torch 

class Generate_Response_Prob:
    def __init__(
            self,
            tokenizer, 
            num_responses_for_one_prompt= 1
            batched_response = 1 
            ):
        self.tokenizer = tokenizer
        self.num_responses = num_responses_for_one_prompt
        self.batched_response = batched_response 

    def compute_prob( self,model, prompt, generated_response_k ):
        log_prob_list = []
        log_prob = 0
        for i in range( self.num_responses ):
            generated_response = generated_response_k[i : i+1 ].squeeze(0)

            response_tokenized = self.tokenizer(
                    generated_response,
                    return_tensors = 'pt', 
                    truncation = True,
                    padding = 'max_length'
                            ) 
            
            for j in range( tokenizer.max_length_single_sentance):
                prompt_tokenized = self.tokenizer( 
                                                  prompt,
                                                  return_tensors = 'pt', 
                                                  truncation = True, 
                                                  padding = 'max_length'
                                                  ) 
                
                model_output = model( **prompt_tokenized ).logits 

                soft_max_model_output = torch.nn.functional.softmax(model_output[:,-1,:], dim = -1)
                log_prob = log_prob + torch.log(soft_max_model_output[ :,response_tokenized[:,:, j ] ] 
                prompt = prompt + self.tokenizer.decode( tokenized_ids )

            log_prob_list.append( log_prob ) 
        
        return torch.tensor(
                log_prob_list, 
                requires_grad = True 
                )



