from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os
import pickle

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
print("Current Working Directory:", os.getcwd())

def read_scens(file_path):
    scens = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            # 去除可能存在的多余空白字符，如换行符
            line = line.strip()
            if not line:  # 如果是空行则跳过
                continue

            # 分割任务名称和描述，假设使用逗号加单引号作为分隔符
            parts = line.split("','")
            if len(parts) != 2:
                print(f"Warning: Line {index} does not follow the expected format and will be skipped.")
                continue

            scen_name = parts[0].strip("'")  # 移除任务名称两端的单引号
            description = parts[1].strip("'\n")  # 移除描述两端的单引号及可能存在的末尾换行符

            # 创建字典并添加到列表中
            scen_info = {
                "scen_id": index,
                "scen_name": scen_name,
                "description": description
            }
            scens.append(scen_info)

    return scens


# 使用示例
file_path = 'scen_and_description'  # 替换为你的文件路径
scens_list = read_scens(file_path)

def read_skills_and_scens(file_path):
    skills_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除行首尾的空白字符，包括换行符
            line = line.strip()

            # 如果是空行则跳过
            if not line:
                continue

            # 移除所有空格和'-'符号
            line = line.replace(' ', '').replace('-', '')

            # 分割技能名称和任务名称
            parts = line.split(',')
            if len(parts) < 2:
                print(f"Warning: Line does not contain both skill and scens and will be skipped.")
                continue

            skill_name = parts[0]
            scen_names = parts[1:]

            # 将技能名称作为键，对应的任务列表作为值加入到字典中
            skills_dict[skill_name] = scen_names

    return skills_dict


# 使用示例
file_path = 'skill_and_scen'  # 替换为你的文件路径
skills_scens_dict = read_skills_and_scens(file_path)

## 检查任务属于哪些技能
cnt = 0
muti_relation_scens = []

for i in range(len(scens_list)):
    scen = scens_list[i]['scen_name']
    found_flag = 0
    for skill, scens in skills_scens_dict.items():
        if scen in scens:
            found_flag = found_flag + 1
            cnt = cnt + 1
            print(f'{scen} belongs to {skill}.')
    if found_flag >= 2:
        muti_relation_scens.append((scen,found_flag))
    if not found_flag:
        print(f'xxx{scen} (id {i}) not belongs to any skill.xxx')

print(f'there are {len(scens_list)} scens and {cnt} belonging relations.')
print(f'muti relation scens: {muti_relation_scens}')
# print(len(muti_relation_scens))


# print('scens_list:')
# print(scens_list)
#
# print('skills_scens_dict:')
# print(skills_scens_dict)

# 加载预训练模型和分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def get_embedding(text):
    """获取文本的RoBERTa嵌入向量"""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def process_scenarios(scens_list, skills_dict, output_file='scens_embeddings.npy'):
    """
    处理情景数据并生成编码后结果
    Args:
        scens_list: 原始情景列表
        skills_dict: 技能-情景映射字典
        output_file: 嵌入向量输出文件
    Returns:
        enco_scens_list: 包含编码信息的新情景列表
    """
    enco_scens_list = []
    embeddings_list = []

    # 逐条处理情景数据
    for scen in scens_list:
        # 获取描述嵌入
        embedding = get_embedding(scen['description'])

        # 收集相关技能
        current_skills = []
        for skill, scen_list in skills_dict.items():
            if scen['scen_name'] in scen_list:
                current_skills.append(skill)

        # 构建新情景条目
        new_entry = {
            **scen,
            'skill': current_skills,
            'enco_description': embedding
        }
        enco_scens_list.append(new_entry)

        # 收集嵌入向量
        embeddings_list.append(embedding)

    # 保存嵌入向量
    embeddings_array = np.vstack(embeddings_list)
    np.save(output_file, embeddings_array)

    return enco_scens_list

# 处理数据
print('开始处理')

result_list = process_scenarios(scens_list, skills_scens_dict)

print(f"成功处理{len(result_list)}条情景数据")
print(f"嵌入向量文件已保存至：scens_embeddings.npy")

print(result_list[18])

with open('scen_skill_desc_list.pkl', 'wb') as f:
    pickle.dump(result_list, f)

print('完成pickle保存')