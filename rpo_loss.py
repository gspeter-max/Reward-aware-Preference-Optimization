import torch 

class rpo_loss_func:
        def __init__( self, ref_model , policy_model , beta, distance_matrix ):
        self.ref_model = ref_model 
        self.policy_model = policy_model 
        self.beta = beta 
        self.D = distance_matrix 
    
    def backward_kl( self,y,x):
        '''
        first we are need to compute π(yk∣x) likehood of the model to generate a prompt 
        that is done by computing the sum of all the token probability that have in respone 
        '''

        model_respone_likelihood = 
        exp_response_term = torch.exp( self.beta * torch.log(

    def __call__(self , 

