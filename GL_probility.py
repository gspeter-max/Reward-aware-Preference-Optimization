import torch 

class Generate_Response_Prob:
    def __init__(
            self, 
            model, 
            tokenizer
            ):
        self.model = model
        self.tokenizer = tokenizer 

    def compute_prob( self, prompt, generated_response ):
        log_prob = 0 
        response_tokenized = tokenizer( generated_response )

        for index , tokenized_ids in enumerate(response_tokenized.input_ids):
            prompt_tokenized = self.tokenizer( prompt, return_tensors = 'pt' )
            model_output = model( **prompt_tokenized ).logits 
            soft_max_model_output = torch.nn.functional.softmax(model_output[:,-1,:], dim = -1)
            log_prob = log_prob + torch.log(soft_max_model_output[ :,tokenized_ids])
            prompt = prompt + self.tokenizer.decode( tokenized_ids )
            print(prompt) 
        
        return log_prob 


