import time

from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 加载预训练的 RoBERTa 分词器和模型
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# 如果需要 GPU 加速
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# text = "这是一个示例句子。"
# # 分词并转换为张量
# inputs = tokenizer(
#     text,
#     return_tensors='pt',        # 返回 PyTorch 张量
#     padding=True,               # 自动填充
#     truncation=True,            # 自动截断
#     max_length=512              # 最大长度
# ).to(device)                   # 移动到 GPU
#
# with torch.no_grad():
#     outputs = model(**inputs)
#
# # 获取最后一层的输出（shape: [batch_size, sequence_length, hidden_size=768]）
# last_hidden_states = outputs.last_hidden_state
#
# cls_embedding = last_hidden_states[:, 0, :]  # shape: [1, 768]

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

emb1 = get_embedding("follow the red car in front of you.")
emb2 = get_embedding("follow the black car in front of you.")
emb3 = get_embedding("park the car.")
similarity1 = torch.nn.functional.cosine_similarity(torch.tensor(emb1), torch.tensor(emb2))
print(similarity1.item())
similarity2 = torch.nn.functional.cosine_similarity(torch.tensor(emb1), torch.tensor(emb3))
print(similarity2.item())
print(round(similarity2.item()/similarity1.item(),3))

def get_embeddings(texts):
    outputs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = model(**inputs)
        outputs.append(output.last_hidden_state[:, 0, :].cpu().numpy())
    print(f'outputs shape: {np.array(outputs).squeeze(axis=1).shape}')
    return np.array(outputs).squeeze(axis=1)

# texts = [
#     'The vehicle automatically identifies a parking space and parks in, avoiding obstacles around.',
#     'The system locates a suitable spot and performs precise parking for different types of spaces.',
#     'Sensors help the vehicle judge available space to ensure safe and accurate parking.',
#
#     'Adjusts speed based on the leading vehicle to maintain a safe distance and avoid collision.',
#     'Keeps stable following behavior by sensing changes in front vehicle movement.',
#
#     'Smoothly merges into the target lane without affecting surrounding vehicles.',
#     'Evaluates traffic conditions and selects the right time to change lanes.',
#     'Predicts other drivers’ behavior to ensure safe and seamless merging.',
# ]
#
# time_start = time.time()
# embeddings = get_embeddings(texts)
# print(f'embed spend: {(time.time()-time_start)/len(texts)} secs per text.')
#
# tsne = TSNE(perplexity=3, n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embeddings)
#
# # 定义组别和对应的颜色
# groups = [0]*3 + [1]*2 + [2]*3  # 第一组前三个，第二组中间两个，第三组最后三个
# colors = ['red', 'green', 'blue']  # 每个组的基础颜色
#
# # 绘制散点图
# plt.figure(figsize=(10, 8))
# for idx, group in enumerate(set(groups)):
#     mask = [g == group for g in groups]
#     plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
#                 label=f'Group {group+1}', color=colors[idx], alpha=0.7)
#
# # 添加注释（可选）
# for i, text in enumerate(texts):
#     plt.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=9, alpha=0.7)
#
# plt.legend()
# plt.title('t-SNE Visualization of Text Embeddings')
# plt.show()
#
# # print(embeddings[0].shape, torch.tensor(embeddings[1]).unsqueeze(0).shape)
# # print('\n')
#
# similarity = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[0]).unsqueeze(0), torch.tensor(embeddings[1]).unsqueeze(0))
# print(similarity.item())
# similarity = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[3]).unsqueeze(0), torch.tensor(embeddings[4]).unsqueeze(0))
# print(similarity.item())
# similarity = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[5]).unsqueeze(0), torch.tensor(embeddings[6]).unsqueeze(0))
# print(similarity.item())
#
# similarity = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[0]).unsqueeze(0), torch.tensor(embeddings[3]).unsqueeze(0))
# print(similarity.item())
# similarity = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[0]).unsqueeze(0), torch.tensor(embeddings[5]).unsqueeze(0))
# print(similarity.item())
# similarity = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[3]).unsqueeze(0), torch.tensor(embeddings[5]).unsqueeze(0))
# print(similarity.item())

texts = ['Reach a goal position. Randomize the goal positions',
         'Push the puck to a goal. Randomize puck and goal positions',
         'Pick and place a puck to a goal. Randomize puck and goal positions',
         'Open a door with a revolving joint. Randomize door positions',
         'Rotate the faucet counter-clockwise. Randomize faucet positions',
         'Push and close a drawer. Randomize the drawer positions',
         'Press a button from the top. Randomize button positions',
         'Unplug a peg sideways. Randomize peg positions',
         'Push and open a window. Randomize window positions',
         'Push and close a window. Randomize window positions']

time_start = time.time()
embeddings = get_embeddings(texts)
print(f'embed spend: {(time.time()-time_start)/len(texts)} secs per text.')

for i in range(len(texts)):
    if i != len(texts)-1:
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[i]).unsqueeze(0), torch.tensor(embeddings[i+1]).unsqueeze(0))
        print(similarity.item())
    else:
        similarity = torch.nn.functional.cosine_similarity(torch.tensor(embeddings[i]).unsqueeze(0), torch.tensor(embeddings[0]).unsqueeze(0))
        print(similarity.item())