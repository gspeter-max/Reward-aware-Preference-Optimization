from datasets import Dataset

dataset = Dataset.from_list([
    {
        "prompt": "What are the benefits of AI in healthcare?",
        "responses": [
            "AI helps diagnose diseases earlier.",
            "It can reduce hospital readmission rates.",
            "AI assists doctors in analyzing medical images.",
            "Chatbots can support mental health therapy."
        ]
    },
    {
        "prompt": "Explain the concept of overfitting in machine learning.",
        "responses": [
            "Overfitting is when a model performs well on training data but poorly on test data.",
            "It happens when the model learns noise instead of the actual pattern.",
            "A sign of overfitting is high training accuracy and low validation accuracy.",
            "Regularization techniques can help prevent overfitting."
        ]
    },
    {
        "prompt": "What causes inflation in an economy?",
        "responses": [
            "An increase in consumer demand can drive inflation.",
            "Printing too much money leads to currency devaluation.",
            "Supply chain disruptions can increase production costs.",
            "Wage hikes without productivity gains may fuel inflation."
        ]
    }
])
print(dataset['responses'])
