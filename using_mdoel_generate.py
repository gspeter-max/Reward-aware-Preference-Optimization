import torch

def compute_score( model_input : dict ,model_name = None , model = None ,tokenizer = None, for_each_token = False):
    from transformers import AutoModelForCausalLM

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model is None :
        model = AutoModelForCausalLM.from_pretrained( model_name ).to(device)
    model_input = next(iter(model_input))

    with torch.no_grad():
        output = model( **model_input )
        logits = output.logits
    print(f'logits shape : {logits.shape}')
    '''
    logits --> ( Batch , SeqLen , Vocab_size )
    log_softmax ----> ( Batch, SeqLen , Vocab_size )

    '''

    log_softmax = torch.nn.functional.log_softmax(logits, dim = -1 )
    seq_scores = []
    i = 1 
    for input_id in range(model_input['input_ids'].shape[2]):
        if i ==1 :
            print(log_softmax[:, :,input_id - 1 , input_id].shape)
            i +=1 
        seq_scores.append( log_softmax[:, :,input_id - 1 , input_id ] )
    if i == 2:
        print(torch.stack(seq_scores, dim = -1).shape) 

    if for_each_token :
        return torch.tensor(seq_scores)
    scores = torch.mean(torch.stack(seq_scores, dim = -1),-1).to(torch.float32)
    scores = (scores - scores.min())/ (scores.max() - scores.min()) 
    return scores

print(compute_score(model_input=dataset, model_name='gpt2'))

