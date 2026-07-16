def read_tasks(file_path):
    tasks = []
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

            task_name = parts[0].strip("'")  # 移除任务名称两端的单引号
            description = parts[1].strip("'\n")  # 移除描述两端的单引号及可能存在的末尾换行符

            # 创建字典并添加到列表中
            task_info = {
                "task_id": index,
                "task_name": task_name,
                "description": description
            }
            tasks.append(task_info)

    return tasks


# 使用示例
file_path = 'task_and_description'  # 替换为你的文件路径
tasks_list = read_tasks(file_path)

# 打印结果以验证正确性
print(f'tasks_list:')
print(tasks_list)
# for task in tasks_list:
#     print(task)