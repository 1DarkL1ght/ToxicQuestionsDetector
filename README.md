# Toxic questions detector
## Description
Here's simple API for my language model that detects toxic questions in english
## Train data
I used [Quora Insincere Questions Classification](https://www.kaggle.com/competitions/quora-insincere-questions-classification), [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data) datasets and 
[GloVe embeddings](https://www.kaggle.com/datasets/takuok/glove840b300dtxt/code)
## Model architecture
The model consists of 2 bidirectional LSTM layers with hidden size of 256, followed by MultiHeadAttention and linear layer(hidden_size * 2, 1)
## Data for fine-tuning and training
You need to provide Pandas DataFrame with columns 'question_text' and 'target'
- Question_text: texts of the questions
- Target: 1 for toxic question, 0 for neutral
## Future plans
- Try different architectures, such as [Transformer](https://arxiv.org/abs/1706.03762), [Attention based LSTM](https://medium.com/@eugenesh4work/attention-mechanism-for-lstm-used-in-a-sequence-to-sequence-task-be1d54919876) (done), [Encoder-only Transformer](https://medium.com/@RobuRishabh/types-of-transformer-model-1b52381fa719)
- Extend API with methods for fine-tuning and training model
- Export model to ONNX format
- Extend dataset
