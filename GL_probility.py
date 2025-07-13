import torch 

class Generate_Response_Prob:
    def __init__(
            self,
            tokenizer
            ):
        self.tokenizer = tokenizer 

    def compute_prob( self,model, prompt, generated_response_k ):
        log_prob_list = [] 
        log_prob = 0
        for generated_response in generated_response_k:
            response_tokenized = tokenizer( generated_response )
            for index , tokenized_ids in enumerate(response_tokenized.input_ids):
                prompt_tokenized = self.tokenizer( prompt, return_tensors = 'pt' )
                model_output = model( **prompt_tokenized ).logits 
                soft_max_model_output = torch.nn.functional.softmax(model_output[:,-1,:], dim = -1)
                log_prob = log_prob + torch.log(soft_max_model_output[ :,tokenized_ids])
                prompt = prompt + self.tokenizer.decode( tokenized_ids )
            log_prob_list.append( log_prob ) 
        
        return log_prob_list


