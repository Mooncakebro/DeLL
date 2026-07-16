import os
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


print('scens_list:')
print(scens_list)

print('skills_scens_dict:')
print(skills_scens_dict)