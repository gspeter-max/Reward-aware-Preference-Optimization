import torch

from datasets import Dataset
from torch.utils.data import DataLoader

class get_data:

    def __init__(
            self,
            tokenizer
            ):
        self.tokenizer = tokenizer

    def __call__(self):
        from datasets import Dataset
        dict_ = {
            "prompt": [
                "What are the benefits of AI in healthcare?",
                "Explain the concept of overfitting in machine learning."
                ],
            "responses": [[
                "AI helps diagnose diseases earlier.",
                "It can reduce hospital readmission rates."
            ],
            [
                "Overfitting is when a model performs well on training data but poorly on test data.",
                "It happens when the model learns noise instead of the actual pattern."
            ]
            ]
        }
        dataset = Dataset.from_dict(dict_)
        def map_function(example):
            temp_list = []
            for response in example['responses']:
                temp_list.append(f'prompt : {example["prompt"]} [SEP] response : {response}')
            example['final_combination'] = temp_list
            return example

        dataset = dataset.map(map_function)
        dataset  = dataset.remove_columns(['prompt','responses'])
        def map_tokenizer(example):
            return self.tokenizer(
                text = example['final_combination'],
                truncation = True,
                padding = 'max_length',
                return_tensors = 'pt'
                )

        dataset = dataset.map( map_tokenizer )
        dataset.set_format(
                type = 'torch',
                device = 'cuda'
                )
        dataset = DataLoader(dataset, batch_size = len(dataset))
        return dataset

# from transformers import  AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('gpt2')

# tokenizer.pad_token = tokenizer.eos_token
# data = get_data(tokenizer = tokenizer)
# dataset= data()

