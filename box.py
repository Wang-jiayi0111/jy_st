import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict

reward_v = 'v2.1'  # 奖励函数
method = '/'+ reward_v+"/noEWM"
program='notepad__spl'


def load_ocplx_data(file_dict):
    """
    从多个文件加载OCplx数据
    返回: {算法名称: OCplx值列表}
    """
    algorithm_data = defaultdict(list)

    for algo_name, file_path in file_dict.items():
        with open(file_path, 'r') as f:
            text = f.read()
            pattern = r"Best OCplx: (\d+\.\d+)"
            ocplx_values = [float(match.group(1)) for match in re.finditer(pattern, text)]
            algorithm_data[algo_name] = ocplx_values
    
    return algorithm_data

def generate_combined_boxplot(algorithm_data):
    """
    生成合并的箱型图比较多个算法
    
    参数:
        algorithm_data: 字典 {算法名称: OCplx值列表}
    """
    algorithms = list(algorithm_data.keys())
    data = list(algorithm_data.values())
    
    plt.figure(figsize=(10, 6))
    box = plt.boxplot(data, patch_artist=True, labels=algorithms)
    
    color = 'lightblue'  
    for patch in box['boxes']:
       patch.set_facecolor(color)

    plt.title(f'{program} OCplx boxplot', fontsize=14, pad=20)
    plt.ylabel('Best OCplx', fontsize=12)
    plt.xlabel('Algorithms', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'/home/ps/jy_exp/output/boxplot{method}/png/{program}.png', dpi=300)

    stats = {}
    for algo in algorithms:
        stats[algo] = {
            'min': np.min(algorithm_data[algo]),
            '25%': np.percentile(algorithm_data[algo], 25),
            'median': np.median(algorithm_data[algo]),
            '75%': np.percentile(algorithm_data[algo], 75),
            'max': np.max(algorithm_data[algo]),
            'mean': np.mean(algorithm_data[algo])
        }
    return stats

def write_stats_to_file(stats, file_path):
    """
    将统计信息写入文件
    """
    with open(file_path, 'w') as f:
        f.write(f"program\t\tmethod\t\trange\t\taverage\n")
        for algo, values in stats.items():
            f.write(f"{program}\t{algo}{method}\t[{values['min']:.6f},{values['max']:.6f}]\t{values['mean']:.6f}\n")

if __name__ == "__main__":
    file_dict = {
        'A3C': f'/home/ps/jy_exp/output/{program}{method}/A3C/best_sequence_{reward_v}.txt',
        'DQN':f'/home/ps/jy_exp/output/{program}{method}/DQN/best_sequence_{reward_v}.txt',
        'PPO': f'/home/ps/jy_exp/output/{program}{method}/PPO/best_sequence_{reward_v}.txt',
        'SARSA':f'/home/ps/jy_exp/output/{program}{method}/SARSA/best_sequence_{reward_v}.txt'
    }

    algorithm_data = load_ocplx_data(file_dict)
    stats = generate_combined_boxplot(algorithm_data)
    
    print("\nOCplx统计信息:")
    for algo, values in stats.items():
        print(f"\n{algo}:")
        for key, value in values.items():
            print(f"{key}: {value:.6f}")

    output_file_path = f'/home/ps/jy_exp/output/boxplot/{method}/txt/{program}.txt'
    write_stats_to_file(stats, output_file_path)