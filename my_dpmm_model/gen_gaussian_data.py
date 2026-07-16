import numpy as np
import matplotlib.pyplot as plt


def generate_gaussian_clusters():
    # 定义每个簇的均值和协方差矩阵
    cluster_params = [
        {'mean': [0.0, 0.0], 'cov': [[1, 0], [0, 1]]},  # 簇0
        {'mean': [0.5, 0.5], 'cov': [[1, 0], [0, 1]]},  # 簇1（接近簇0）
        {'mean': [3.0, 3.0], 'cov': [[1, 0], [0, 1]]},  # 簇2（靠近簇0）
        {'mean': [8.0, 8.0], 'cov': [[0.5, 0], [0, 0.5]]},  # 簇3（远离其他簇）
        {'mean': [0.0, 7.0], 'cov': [[0.5, 0], [0, 0.5]]}  # 簇4（远离其他簇）
    ]

    data = []
    for i, params in enumerate(cluster_params):
        samples = np.random.multivariate_normal(
            mean=params['mean'],
            cov=params['cov'],
            size=100
        )
        data.append({
            'id': i,
            'obs': samples  # shape: (100, 2)
        })

    return data


def plot_clusters(data):
    # 设置颜色
    colors = ['#1f77b4', '#6baed6', '#bdd7e7', '#d62728', '#fc8d62']  # 前三个为蓝色系，后两个为红色系

    plt.figure(figsize=(8, 6))

    for i, cluster in enumerate(data):
        points = cluster['obs']
        plt.scatter(points[:, 0], points[:, 1], c=colors[i], label=f'Cluster {i}', s=30)

    # 可选：画出各簇中心
    centers = np.array([cluster['obs'].mean(axis=0) for cluster in data])
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, marker='x', label='Centers')

    plt.title('2D Gaussian Clusters')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # 保证坐标轴比例一致
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    data = generate_gaussian_clusters()
    plot_clusters(data)


