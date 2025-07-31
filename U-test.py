import numpy as np
from scipy.stats import mannwhitneyu

# 生成示例数据（替换为你的实际数据）
group1 = np.array([1, 1, 1, 1, 1, 1, 1, 1])
group2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])

# 执行Wilcoxon秩和检验
statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

# 输出结果
print(f"Mann-Whitney U statistic: {statistic:.4f}")
print(f"P-value: {p_value:.6f}")

# 根据p值解释结果
alpha = 0.05
if p_value < alpha:
    print("Reject null hypothesis: Significant difference between groups")
else:
    print("Fail to reject null hypothesis: No significant difference")