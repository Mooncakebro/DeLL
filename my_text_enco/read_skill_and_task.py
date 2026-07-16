def read_skills_and_tasks(file_path):
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
                print(f"Warning: Line does not contain both skill and tasks and will be skipped.")
                continue

            skill_name = parts[0]
            task_names = parts[1:]

            # 将技能名称作为键，对应的任务列表作为值加入到字典中
            skills_dict[skill_name] = task_names

    return skills_dict


# 使用示例
file_path = 'skill_and_task'  # 替换为你的文件路径
skills_tasks_dict = read_skills_and_tasks(file_path)

# 打印结果以验证正确性
for skill, tasks in skills_tasks_dict.items():
    print(f"Skill: {skill}")
    print("Tasks:", tasks)