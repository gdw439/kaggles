from torch.optim import Adam
from tqdm import tqdm
import re
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained('./model')
# model = BertModel.from_pretrained("./model")

# text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# print(encoded_input)
# output = model(**encoded_input)
# print(output)

tokenizer = BertTokenizer.from_pretrained('./model')

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.labels = [label for label in df['target']] if 'target' in df else [0 for i in range(len(df['text']))]
        temp = []
        for i, text in enumerate(df['text']):
            perfix = ""
            if isinstance(df['keyword'][i], str):
                perfix += 'label: ' + df['keyword'][i]
            
            if isinstance(df['location'][i], str):
                perfix += ', location: ' + df['location'][i]

            text = perfix + text
            text = re.sub(r'@[A-Za-z0-9_]{,32}', " ", text)
            text = re.sub(r'http://t\.co/[A-Za-z0-9]{,10}', " ", text)
            text = re.sub(r'https://t\.co/[A-Za-z0-9]{,10}', " ", text)
            temp.append(text)
        
        # import codecs
        # h = codecs.open('show.txt', 'w', encoding='utf-8')
        # for t in temp:
        #     h.write(t + '\n')
        # h.close()
            
        self.texts = [tokenizer(text, 
            padding='max_length', 
            max_length = 512, 
            truncation=True,
            return_tensors="pt") 
        for text in temp]


    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('./model_tune')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer


def test(model, train_data, val_data, learning_rate, epochs):
    # 通过Dataset类获取训练和验证集
    test, val = Dataset(train_data), Dataset(val_data)
    # DataLoader根据batch_size获取数据，训练时选择打乱样本
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=64)
    # 判断是否使用GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    # 开始进入训练循环
    for epoch_num in range(epochs):
        # 定义两个变量，用于存储训练集的准确率和损失
        total_acc_train = 0
        total_loss_train = 0
        # 进度条函数tqdm
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            # 通过模型得到输出
            output = model(input_id, mask)
            # 计算损失
            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()
            # 计算精度
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            # 模型更新
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # ------ 验证模型 -----------
            # 定义两个变量，用于存储验证集的准确率和损失
            total_acc_val = 0
            total_loss_val = 0
            # 不需要计算梯度
            with torch.no_grad():
                # 循环获取数据集，并用训练好的模型进行验证
                for val_input, val_label in val_dataloader:
                # 如果有GPU，则使用GPU，接下来的操作同训练
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)
  
                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label)
                    total_loss_val += batch_loss.item()
                    
                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
            # torch.save(model, 'tune.bin') 
            print(
                f'''Epochs: {epoch_num + 1} 
              | Train Loss: {total_loss_train / len(train_data): .3f} 
              | Train Accuracy: {total_acc_train / len(train_data): .3f} 
              | Val Loss: {total_loss_val / len(val_data): .3f} 
              | Val Accuracy: {total_acc_val / len(val_data): .3f}''')     
    torch.save(model, 'tune.bin')


def evaluate(model, test_data):
    test = Dataset(test_data)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=64)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    
    label = []
    total_acc_test = 0
    with torch.no_grad():
        for test_input, _ in test_dataloader:
            # test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            
            # print(output.argmax(dim=1))
            label.extend(output.argmax(dim=1).tolist())
    test_data['target'] = label


import pandas as pd
df_test = pd.read_csv('./dataset/test.csv')
model = torch.load('./tuned/tune.bin')
evaluate(model, df_test)
df_test.to_csv('test.csv')
