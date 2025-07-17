# from IPython import get_ipython
# if (get_ipython().__class__.__name__).lower() == 'shell':
#     ! pip install -U datasets

# else :
    # pip install -U datasets

from datasets import Dataset
dict_ = {
        "prompt": ["What are the benefits of AI in healthcare?", "Explain the concept of overfitting in machine learning.","What causes inflation in an economy?"],
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

print(dataset['responses'])
