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
                "Explain the concept of overfitting in machine learning.",
                "What causes inflation in an economy?"
                ],
            "responses": [[
                "AI helps diagnose diseases earlier.",
                "It can reduce hospital readmission rates.",
                "AI assists doctors in analyzing medical images.",
                "Chatbots can support mental health therapy."
            ],
            [
                "Overfitting is when a model performs well on training data but poorly on test data.",
                "It happens when the model learns noise instead of the actual pattern.",
                "A sign of overfitting is high training accuracy and low validation accuracy.",
                "Regularization techniques can help prevent overfitting."
            ],
            [
                "An increase in consumer demand can drive inflation.",
                "Printing too much money leads to currency devaluation.",
                "Supply chain disruptions can increase production costs.",
                "Wage hikes without productivity gains may fuel inflation."
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

        dataset = dataset.map(map_function, remove_columns = dataset.column_names)

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
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                )
        dataset = dataset.select_columns(['input_ids','attention_mask'])
        dataset = DataLoader(dataset, batch_size = len(dataset))
        return dataset

from transformers import  AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

data = get_data(tokenizer = tokenizer)
dataset= data()
