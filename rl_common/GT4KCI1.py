import os
import re
import json
import torch
import random
import logging
import platform
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from node2vec import Node2Vec
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_curve, average_precision_score
from pathlib import Path
from collections import Counter

BASE_PATH = "/home/ps/jy_exp/input_GNN"
# 新数据集名称
NEW_DATASET = "gt4kciProject"
# 采样比例（每个数据集取20%）
SAMPLE_RATIO = 1.0
# 要组合的数据集列表
DATASETS = ["JHotDraw"]

create_hybrid = True  # 设为True时创建混合数据集
# 是否使用真实的数据
is_real_data = True
# 数据集名称
if create_hybrid:
    dataset = "Hybrid"
else:
    dataset = "SPM"
# 网络类型 MCN(Multilayer Class Network) MPN(Multilayer Package Network)
net_type = "MCN"
# 数据路径  
real_data_path = f"/home/ps/jy_exp/input_GNN/{dataset}_{net_type}/combined_{dataset}_GN.net"
# 是否进行参数调优的
searching_best_param = False
# 关键类占比
key_class_percentage = 0.2
# 关键类数量
num_key_nodes = 10
# 是否使用预定义关键类 如果为True则上面的num_key_nodes设置不会生效
use_predefined_key_classes = False
# 找不到精确匹配时是否允许模糊匹配
allow_fuzzy_matching = True
# 没有找到的关键类是否通过中心性计算来补充
auto_complete_missing_classes = True
# 是否使用上次训练的模型继续训练
use_previous_model = True
# 是否启用迁移学习模式
enable_transfer_learning = True
# 是否标准化节点特征
standardize_features = True
# 是否使用多种中心性指标作为特征
use_multiple_centrality = True
# 迁移学习设置
transfer_learning_settings = {
    'adaptive_learning_rate': True,    # 自适应学习率
    'gradual_unfreezing': True,        # 渐进式解冻
    'feature_adaptation': True,        # 特征适应层
    'feature_weighting': True          # 特征权重
}
# 预定义的关键类（与use_predefined_key_classes相关）
predefined_key_classes = {
    "ANT": ["ElementHandler", "IntrospectionHelper", "Main", "Project", "ProjectHelper",
            "RuntimeConfigurable", "Target", "Task", "TaskContainer", "UnknownElement"],
    "Hibernate": ["Column", "Configuration", "ConnectionProvider", "Criteria", "Criterion", "Projection", 
                  "Query", "Session", "SessionFactory", "SessionFactoryImplementor", "SessionImplementor", 
                  "Table", "Transaction", "Type"],
    "jEdit": ["Buffer", "EBMessage", "EditPane", "jEdit", "JEditTextArea", "Log", "View"],
    "JGAP": ["AveragingCrossoverOperator", "BaseGene", "BaseGeneticOperator", "BestChromosomesSelector", 
             "BooleanGene", "Chromosome", "Configuration", "CrossoverOperator", "DoubleGene", 
             "FitnessFunction", "FixedBinaryGene", "GaussianMutationOperator", "Genotype", 
             "IntegerGene", "MutationOperator", "NaturalSelector", "Population", "WeightedRouletteSelector"],
    "JHotDraw": ["CompositeFigure", "Drawing", "DrawingEditor", 
                 "Figure", "Handle", "StandardDrawingView", "DrawApplication", "DrawingView", "Tool"],
    "JMeter": ["AbstractAction", "JMeterEngine", "JMeterGUIComponent", "JMeterThread", "JMeterTreeModel", 
               "PreCompiler", "Sampler", "SampleResult", "TestCompiler", "TestElement", "TestListener", 
               "TestPlan", "TestPlanGui", "ThreadGroup"],
    "Log4j": ["Appender", "Configuration", "Filter", "Layout", "Logger", "LoggerConfig", 
              "LoggerContext", "StrLookup", "StrSubstitutor"],
    "Wro4J": ["Group", "Resource", "ResourcePostProcessor", "ResourcePreProcessor", "ResourceType", 
              "UriLocator", "UriLocatorFactory", "WroFilter", "WroManager", "WroManagerFactory", 
              "WroModel", "WroModelFactory"]
}

# 训练信息输出频率
output_frequency = 5
# 预设的最佳参数
preset_best_param = {
    'learning_rate': 1e-3,  # 学习率
    'num_sampled_neighbors': 10,# 20
    'num_layers': 1,            
    'dropout_rate': 0.1,        # 0.3
    'weight_decay': 0.0005,  # L2正则化参数
    'batch_norm': True
}
num_epochs = 0  # 训练轮数

# Node2Vec 的 p 和 q 参数 和 纬度参数
node2vec_p_param = 0.25    # 修改为0.5，更好的平衡BFS和DFS
node2vec_q_param = 2    # 修改为1.0，使用平衡的策略
Node2Vec_dimensions = 32  # 增加到64维，提供更丰富的特征表示        维度匹配不上的原因
# 中心性调整的参数
ramdom_gama_beta = False
GKCI_gama = 6.2395
GKCI_beta = -13.8278

# 早停参数
early_stopping_patience = 1000
early_stopping_delta = 0.001

# 数据集划分次数
num_splits = 1

# -----------------------------------
# 日志配置
# -----------------------------------
# 确保结果目录存在
result_dir = "/home/ps/jy_exp/output/GNN_res4"
os.makedirs(result_dir, exist_ok=True)

# 历史模型保存路径
historical_models_dir = "/home/ps/jy_exp/output/GNN_res4/historicalModels"
os.makedirs(historical_models_dir, exist_ok=True)

# 获取下一个结果文件的序号，命名为 result-1.txt, result-2.txt, ...
def get_next_result_file():
    log_dir = os.path.join(result_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    existing_files = os.listdir(log_dir)
    pattern = re.compile(r'log-(\d+)\.txt')
    max_num = 0
    for filename in existing_files:
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return os.path.join(log_dir, f"log-{max_num + 1}.txt"), max_num + 1


result_file_path, train_id = get_next_result_file()


# 配置日志
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# 创建格式器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# 创建文件处理器
file_handler = logging.FileHandler(result_file_path, mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# -----------------------------------
# 1. 数据处理
# -----------------------------------

# 类依赖网络建模  构建有向图G和ID到名称的映射（G, node_mapping）
def read_net_file(file_path):
    """
    读取并解析 .net 文件，返回一个有向加权图和节点编号到名称的映射。

    :param file_path: .net 文件路径
    :return: (NetworkX DiGraph, dict) 图和节点编号到名称的映射
    """
    G = nx.DiGraph()
    node_mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    vertices_section = False
    arcs_section = False
    for line in lines:
        line = line.strip()
        if line.startswith('*Vertices'):
            vertices_section = True
            arcs_section = False
            num_vertices = int(line.split()[1])
            continue
        elif line.startswith('*Arcs'):
            vertices_section = False
            arcs_section = True
            continue
        elif vertices_section:
            # 使用正则表达式提取节点编号和名称
            match = re.match(r'(\d+)\s+"(.+)"', line)
            if match:
                node_id = int(match.group(1)) - 1  # 假设节点编号从1开始，转换为从0开始
                node_name = match.group(2)
                G.add_node(node_id, name=node_name)
                node_mapping[node_id] = node_name
        elif arcs_section:
            # 提取边信息
            parts = line.split()
            if len(parts) == 3:
                source = int(parts[0]) - 1  # 转换为从0开始
                target = int(parts[1]) - 1
                weight = float(parts[2])
                G.add_edge(source, target, weight=weight)
    return G, node_mapping


# 1.1 生成类依赖网络    根据数据集-和上面无区别&自生样例（G, node_mapping）
def generate_sample_graph():
    """
    生成一个样例类依赖网络（有向加权图）。
    每个节点代表一个类，边代表类之间的依赖关系，权重表示依赖强度。
    """
    if create_hybrid:
        G, node_mapping = read_net_file(combined_net_file)
        return G, node_mapping

    if is_real_data:
        G, node_mapping = read_net_file(real_data_path)
        return G, node_mapping

    G = nx.DiGraph()
    num_nodes = 20
    G.add_nodes_from(range(num_nodes))

    # 添加有向边及其权重
    edges = [
        (0, 1, 3), (0, 2, 1), (0, 3, 2),
        (1, 4, 2), (1, 5, 3),
        (2, 5, 2), (2, 6, 4),
        (3, 6, 1), (3, 7, 3),
        (4, 8, 2), (5, 8, 1), (5, 9, 4),
        (6, 9, 2), (6, 10, 3),
        (7, 10, 1), (7, 11, 4),
        (8, 12, 2), (9, 12, 3), (10, 13, 2),
        (11, 13, 1), (11, 14, 3),
        (12, 15, 4), (13, 15, 2), (14, 15, 1),
        (15, 16, 3), (16, 17, 2),
        (17, 18, 1), (18, 19, 5),
        (19, 0, 2),  # 添加环路
        (4, 2, 1), (5, 3, 2),
        (7, 4, 1), (10, 5, 2),
        (12, 7, 3), (15, 10, 2),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    return G, None


# 获取关键类标签  返回标签数组和关键节点列表（labels[], key_nodes[]）
def get_key_class_labels(G, node_mapping=None):
    """
    根据度中心性和介数中心性计算每个节点的综合中心性，
    或者使用预定义的关键类，并选择相应节点作为关键类。

    :param G: NetworkX 图
    :param node_mapping: 节点ID到节点名称的映射
    :return: 标签数组（1表示关键类，0表示非关键类），关键类节点列表
    """
    # 创建标签数组
    labels = np.zeros(len(G.nodes()))
    # # 如果是混合数据集，从文件加载标签
    # if dataset == "Hybrid":
    #     label_path = real_data_path.replace(".net", "_labels.json")
    #     if os.path.exists(label_path):
    #         with open(label_path, 'r') as f:
    #             data = json.load(f)
    #             labels = data['labels']
    #             # 转换为numpy数组
    #             labels = np.array(labels, dtype=int)
    #             # 获取关键节点列表
    #             key_nodes = [node for node, label in enumerate(labels) if label == 1]
    #             return labels, key_nodes
    # 使用预定义的关键类
    if use_predefined_key_classes and dataset in predefined_key_classes and node_mapping:
        key_class_names = predefined_key_classes[dataset]
        key_nodes = []
        not_found_classes = []
        
        # 创建一个名称到节点ID的映射
        name_to_node = {name: node_id for node_id, name in node_mapping.items()}
        
        # 找到对应的节点ID - 更精确的匹配
        for class_name in key_class_names:
            matched = False
            for name, node_id in name_to_node.items():
                # 获取完整类名的最后一部分（不含包名）
                short_name = name.split('.')[-1]
                # 精确匹配类名
                if short_name == class_name:
                    key_nodes.append(node_id)
                    matched = True
                    logging.info(f"精确匹配找到类 {class_name} => {name}")
                    break
            
            if not matched:
                if allow_fuzzy_matching:
                    # 如果找不到精确匹配，尝试模糊匹配
                    logging.info(f"无法找到类 {class_name} 的精确匹配，尝试模糊匹配")
                    for name, node_id in name_to_node.items():
                        short_name = name.split('.')[-1]
                        if class_name in short_name:
                            key_nodes.append(node_id)
                            logging.info(f"使用模糊匹配找到 {class_name} => {name}")
                            matched = True
                            break
                
                if not matched:
                    not_found_classes.append(class_name)
                    logging.info(f"无法找到类 {class_name}")
        
        # 如果有未找到的类，并且配置了自动补充，则使用中心性计算来补充
        if not_found_classes and auto_complete_missing_classes:
            logging.info(f"使用中心性计算补充未找到的 {len(not_found_classes)} 个类")
            
            # 计算度中心性和介数中心性
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
            combined_centrality = {node: degree_centrality[node] + betweenness_centrality[node] for node in G.nodes()}
            
            # 按中心性排序
            sorted_nodes = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)
            
            # 选择最重要的节点，但要排除已经选择的关键节点
            additional_nodes = []
            for node, centrality in sorted_nodes:
                if node not in key_nodes:
                    additional_nodes.append(node)
                    logging.info(f"自动补充添加节点 {node} => {node_mapping[node]}, 中心性: {centrality:.4f}")
                    if len(additional_nodes) == len(not_found_classes):
                        break
            
            key_nodes.extend(additional_nodes)
        
        # 去重
        key_nodes = list(set(key_nodes))
        
        # 设置关键类标签
        for node in key_nodes:
            labels[node] = 1
            
        logging.info(f"使用预定义的关键类: {key_class_names}")
        logging.info(f"实际找到的关键类数量: {len(key_nodes)}")
        if not_found_classes:
            logging.info(f"未找到的关键类: {not_found_classes}")

    else:
        # 计算度中心性
        degree_centrality = nx.degree_centrality(G)
        # 计算介数中心性，考虑边的权重
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight')

        # 结合度中心性和介数中心性
        combined_centrality = {node: degree_centrality[node] + betweenness_centrality[node] for node in G.nodes()}

        # 选择前20%的节点作为关键类
        # num_key_nodes = max(1, int(key_class_percentage * len(G.nodes())))
        key_nodes = sorted(combined_centrality.items(), key=lambda x: x[1], reverse=True)[:num_key_nodes]
        key_nodes = [node for node, centrality in key_nodes]

        # 设置关键类标签
        for node in key_nodes:
            labels[node] = 1
            
        logging.info(f"使用中心性计算得到的关键类")
    
    # 输出关键类的名称
    print("\n关键类列表:")
    for node in key_nodes:
        if node_mapping:
            print(f"节点ID: {node}, 类名: {node_mapping[node]}")
        else:
            print(f"节点ID: {node}")
    
    return labels, key_nodes


# 1.2 网络嵌入学习  得到节点向量(embeddings)
def compute_node_embeddings(G, dimensions=16):
    """
    使用 Node2Vec 生成节点嵌入向量。

    :param G: NetworkX 图
    :param dimensions: 嵌入维度
    :return: 节点嵌入的 NumPy 数组
    """
    for u, v, data in G.edges(data=True):
        if 'weight' not in data or data['weight'] <= 0:
            print(f"警告: 边({u},{v})有无效权重 {data.get('weight', '无')}")
            data['weight'] = 1.0  # 设置默认权重
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=30,  # 增加游走长度以捕获更多结构信息
        num_walks=100,  # 增加游走次数以提高采样覆盖率
        workers=1,
        p=node2vec_p_param,
        q=node2vec_q_param,
        weight_key='weight',
        quiet=True
    )
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # 增加窗口大小
    
    # 将每个节点的嵌入向量存储为 NumPy 数组
    embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return embeddings


# 计算多种中心性指标    得到G图所有节点的中心性特征(centrality_features)-形状：（节点数, 17）的numpy数组
def compute_centrality_features(G):
    """
    计算图中每个节点的多种中心性指标作为特征。
    增强版本，加入更多中心性指标和特征归一化。
    
    :param G: NetworkX 图
    :return: 节点中心性特征的 NumPy 数组
    """
    logging.info("计算增强的节点中心性特征...")
    
    # 基本中心性指标
    # 度中心性  degree_centrality
    degree_centrality = nx.degree_centrality(G) 
    # 介数中心性    betweenness_centrality
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
    # 尝试使用加权接近中心性    closeness_centrality
    try:
        closeness_centrality = nx.closeness_centrality(G, distance='weight')
    except:
        closeness_centrality = nx.closeness_centrality(G)
    
    # 特征向量中心性    eigenvector_centrality
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except:
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            eigenvector_centrality = nx.pagerank(G, weight='weight')
    
    # 页面排名 - 使用更多迭代次数提高准确性
    pagerank = nx.pagerank(G, weight='weight', max_iter=200, tol=1e-6)
    # 计算加权PageRank变体      pagerank_variants
    alpha_values = [0.7, 0.85, 0.95]
    pagerank_variants = []
    for alpha in alpha_values:
        try:
            pr = nx.pagerank(G, weight='weight', alpha=alpha)
            pagerank_variants.append(pr)
        except:
            # 如果失败，使用默认pagerank
            pagerank_variants.append(pagerank)
    
    # 局部聚类系数  clustering
    try:
        clustering = nx.clustering(G, weight='weight')
    except:
        clustering = nx.clustering(G)
    
    # 计算入度和出度
    in_degree = {node: G.in_degree(node, weight='weight') for node in G.nodes()}
    out_degree = {node: G.out_degree(node, weight='weight') for node in G.nodes()}
    
    # 计算核心部分 (k-core) 指标 - 表示节点在网络中的"核心性"
    try:
        k_core_numbers = nx.core_number(G.to_undirected())
    except:
        # 如果失败，使用基于度的近似
        k_core_numbers = {node: (degree_centrality[node] * 10) for node in G.nodes()}
    
    # 计算HITS算法的hub和authority分数
    try:
        hits_scores = nx.hits(G, max_iter=200)
        hubs, authorities = hits_scores[0], hits_scores[1]
    except:
        # 如果HITS失败，使用度中心性作为近似
        hubs = degree_centrality
        authorities = degree_centrality
    
    # 标准化中心性指标
    if standardize_features:
        logging.info("标准化中心性特征...")
        scaler = StandardScaler()
        centrality_features = []    # 形状为(节点数, 17)的NunPy数值
        
        for node in G.nodes():      # 每个节点具有以下17个特征
            node_features = [
                degree_centrality[node],
                betweenness_centrality[node],
                closeness_centrality[node],
                eigenvector_centrality[node],
                pagerank[node],
                clustering.get(node, 0),
                in_degree[node],
                out_degree[node],
                # 添加新特征
                k_core_numbers.get(node, 0),
                hubs.get(node, 0),
                authorities.get(node, 0),
                # PageRank变体
                pagerank_variants[0].get(node, 0),
                pagerank_variants[1].get(node, 0),
                pagerank_variants[2].get(node, 0),
                # 派生特征
                in_degree[node] / (out_degree[node] + 1e-6),  # 入度/出度比率
                in_degree[node] * out_degree[node],           # 入度和出度的乘积
                in_degree[node] + out_degree[node]            # 入度和出度的和
            ]
            centrality_features.append(node_features)
        
        # 标准化特征
        centrality_features = np.array(centrality_features)
        centrality_features = scaler.fit_transform(centrality_features)
    else:
        # 不标准化，直接构建特征数组
        centrality_features = np.array([
            [
                degree_centrality[node],        # 0.度中心性
                betweenness_centrality[node],   # 1.介数中心性
                closeness_centrality[node],     # 2.接近中心性
                eigenvector_centrality[node],   # 3.特征向量中心性
                pagerank[node],                 # 4.页面排名
                clustering.get(node, 0),        # 5.局部聚类系数
                in_degree[node],                # 6.加权入度
                out_degree[node],               # 7.加权出度
                # 添加新特征
                k_core_numbers.get(node, 0),    # 8.k-core 数量
                hubs.get(node, 0),              # 9.HITS算法的hub分数
                authorities.get(node, 0),       # 10.HITS算法的authority分数
                # PageRank变体
                pagerank_variants[0].get(node, 0),  # 11.PageRank变体1
                pagerank_variants[1].get(node, 0),  # 12.PageRank变体2
                pagerank_variants[2].get(node, 0),  # 13.PageRank变体3
                # 派生特征
                in_degree[node] / (out_degree[node] + 1e-6),  # 14.入度/出度比率
                in_degree[node] * out_degree[node],           # 15.入度和出度的乘积
                in_degree[node] + out_degree[node]            # 16.入度和出度的和
            ] for node in G.nodes()
        ])
    
    # 应用特征重要性权重
    if transfer_learning_settings['feature_weighting']:
        # 为不同类型的特征设置权重
        weights = np.ones(centrality_features.shape[1])
        
        # 结构特征权重 (0-7)
        weights[0:8] = 1.5
        
        # 拓扑特征权重 (8-10)
        weights[8:11] = 1.2
        
        # PageRank变体 (11-13)
        weights[11:14] = 1.3
        
        # 派生特征 (14-16)
        weights[14:17] = 1.1
        
        # 应用权重
        weighted_features = centrality_features * weights
        
        logging.info(f"应用特征重要性权重后的中心性特征形状: {weighted_features.shape}")
        return weighted_features
    
    logging.info(f"增强的中心性特征计算完成，形状: {centrality_features.shape}")
    return centrality_features


# 组合节点嵌入和中心性特征  embeding是否和上面的中心性特征组合（embeddings）
def combine_features(embeddings, centrality_features=None):
    """
    组合节点嵌入和中心性特征。
    
    :param embeddings: 节点嵌入向量
    :param centrality_features: 中心性特征
    :return: 组合后的特征向量
    """
    if centrality_features is not None and use_multiple_centrality:
        # 组合特征
        combined_features = np.hstack((embeddings, centrality_features))
        
        # 标准化组合特征
        if standardize_features:
            scaler = StandardScaler()
            combined_features = scaler.fit_transform(combined_features)
        
        return combined_features
    else:
        # 只使用嵌入特征
        if standardize_features:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)
        
        return embeddings



# -----------------------------------
# 辅助函数：数据集划分（未调用）
# -----------------------------------
def split_data(labels, key_nodes, num_splits=10, test_size=1, random_seed=42):
    """
    将数据集划分为训练集和测试集，确保关键类在两者中均匀分布。

    :param labels: 全部节点的标签数组
    :param key_nodes: 关键类节点列表
    :param num_splits: 划分次数
    :param test_size: 测试集比例（关键类）
    :param random_seed: 随机种子
    :return: 划分后的训练集和测试集标签列表
    """
    splits = []
    for i in range(num_splits):
        # 设置不同的随机种子以保证不同的划分
        seed = random_seed + i
        # 划分关键类
        train_keys, test_keys = train_test_split(key_nodes, test_size=test_size, random_state=seed)

        # 创建训练和测试标签
        train_labels = np.zeros_like(labels)
        test_labels = np.zeros_like(labels)

        # 训练集中的关键类
        for node in train_keys:
            train_labels[node] = 1
        # 测试集中的关键类
        for node in test_keys:
            test_labels[node] = 1

        splits.append((train_labels, test_labels))

    return splits

# 模型保存目录
model_dir = "/home/ps/jy_exp/output/GNN_res/models"
# 训练好的通用模型路径
general_model_path = os.path.join(model_dir, "general_model.pth")

def save_net_file(G, file_path):
    """保存NetworkX图为.net格式"""
    with open(file_path, 'w', encoding='utf-8') as f:
        # 写入节点
        f.write(f"*Vertices {G.number_of_nodes()}\n")
        for node in sorted(G.nodes()):
            name = G.nodes[node].get('name', f'Node{node}')
            f.write(f'{node} "{name}"\n')
        
        # 写入边
        f.write("*Arcs\n")
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            f.write(f"{u} {v} {weight}\n")

def parse_net_file(file_path):
    """解析.net文件为NetworkX图"""
    G = nx.DiGraph()
    node_mapping = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    vertices_section = False
    arcs_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith('*Vertices'):
            vertices_section = True
            arcs_section = False
            continue
            
        if line.startswith('*Arcs'):
            vertices_section = False
            arcs_section = True
            continue
            
        if vertices_section:
            # 解析节点行: "1 \"ClassName\""
            match = re.match(r'(\d+)\s+"(.+)"', line)
            if match:
                node_id = int(match.group(1))
                node_name = match.group(2)
                G.add_node(node_id, name=node_name)
                node_mapping[node_id] = node_name
                
        elif arcs_section:
            # 解析边行: "source target weight"
            parts = line.split()
            if len(parts) >= 2:
                source = int(parts[0])
                target = int(parts[1])
                weight = float(parts[2]) if len(parts) >= 3 else 1.0
                G.add_edge(source, target, weight=weight)
    
    return G, node_mapping

def create_combined_dataset():
    """创建组合数据集"""
    # 1. 创建新数据集目录
    new_dataset_path = Path(BASE_PATH) / f"{NEW_DATASET}_{net_type}"
    new_dataset_path.mkdir(parents=True, exist_ok=True)
    net_file = new_dataset_path / f"combined_{NEW_DATASET}_GN.net"
    
    # 2. 初始化新图的容器
    combined_graph = nx.DiGraph()
    node_id_map = {}  # 存储(原始数据集, 原始ID) -> 新ID
    next_node_id = 1
    
    # 3. 处理每个数据集
    for dataset in DATASETS:
        # 3.1 读取原始数据集
        dataset_path = Path(BASE_PATH) / f"{dataset}_{net_type}"
        source_file = dataset_path / f"combined_{dataset}_GN.net"
        
        if not source_file.exists():
            print(f"⚠️ 跳过缺失的数据集: {source_file}")
            continue
        # 3.2 解析.net文件
        graph, name_map = parse_net_file(source_file)
        
        # 3.3 随机采样节点
        all_nodes = list(graph.nodes())
        sampled_nodes = random.sample(all_nodes, int(len(all_nodes) * SAMPLE_RATIO))
        
        # 3.4 添加采样节点到新图
        for orig_node in sampled_nodes:
            # 创建新节点ID
            new_node_id = next_node_id
            next_node_id += 1
            
            # 保存映射关系
            node_id_map[(dataset, orig_node)] = new_node_id
            
            # 获取类名（添加数据集前缀避免冲突）
            class_name = f"{dataset}_{name_map.get(orig_node, f'Node{orig_node}')}"
            
            # 添加到新图
            combined_graph.add_node(new_node_id, name=class_name)
        
        # 3.5 添加相关边
        for orig_node in sampled_nodes:
            # 获取原始节点的所有出边
            for neighbor in graph.successors(orig_node):
                # 只添加两个端点都被采样的边
                if neighbor in sampled_nodes:
                    weight = graph[orig_node][neighbor].get('weight', 1.0)
                    src = node_id_map[(dataset, orig_node)]
                    dst = node_id_map[(dataset, neighbor)]
                    combined_graph.add_edge(src, dst, weight=weight)
    
    # 4. 保存新数据集
    save_net_file(combined_graph, net_file)
    print(f"✅ 创建组合数据集完成! 共 {combined_graph.number_of_nodes()} 个节点")
    print(f"📁 路径: {net_file}")
    
    return net_file

def compute_centrality_scores(G):
    """
    计算8种中心性指标得分
    :param G: NetworkX 图
    :return: 字典，键为方法名，值为{节点: 得分}的字典
    """
    centrality_scores = {}
    
    # 1. Degree (度中心性)
    centrality_scores['Degree'] = nx.degree_centrality(G)
    
    # 2. In-Degree (入度中心性)
    in_degree = dict(G.in_degree(weight='weight'))
    max_in_degree = max(in_degree.values()) if in_degree else 1
    centrality_scores['In-Degree'] = {node: deg/max_in_degree for node, deg in in_degree.items()}
    
    # 3. PageRank
    centrality_scores['PageRank'] = nx.pagerank(G, weight='weight')
    
    # 4. Betweenness (介数中心性)
    centrality_scores['Betweenness'] = nx.betweenness_centrality(G, weight='weight')
    
    # 5. HITS (使用authority分数)
    try:
        _, authorities = nx.hits(G, max_iter=200)
        centrality_scores['HITS'] = authorities
    except:
        # 如果HITS失败，使用度中心性作为后备
        centrality_scores['HITS'] = nx.degree_centrality(G)
    
    # 6. Cores (k-core)
    try:
        # 转为无向图计算k-core
        undirected_G = G.to_undirected()
        core_numbers = nx.core_number(undirected_G)
        max_core = max(core_numbers.values()) if core_numbers else 1
        centrality_scores['Cores'] = {node: core/max_core for node, core in core_numbers.items()}
    except:
        centrality_scores['Cores'] = nx.degree_centrality(G)
    
    # 7. Weighted K-core (加权核分解)
    try:
        # 使用加权度计算
        weighted_degrees = dict(undirected_G.degree(weight='weight'))
        max_wdegree = max(weighted_degrees.values()) if weighted_degrees else 1
        centrality_scores['Weighted_Kcore'] = {node: deg/max_wdegree for node, deg in weighted_degrees.items()}
    except:
        centrality_scores['Weighted_Kcore'] = nx.degree_centrality(G)
    
    # 8. Minclass (最小类中心性 - 自定义指标)
    # 这里我们使用节点到图中最不重要的节点(度最低)的距离的倒数
    min_degree_node = min(G.degree(), key=lambda x: x[1])[0]
    centrality_scores['Minclass'] = {}
    for node in G.nodes():
        try:
            # 计算到最小度节点的最短路径长度
            path_length = nx.shortest_path_length(G, source=node, target=min_degree_node, weight='weight')
            centrality_scores['Minclass'][node] = 1 / (path_length + 1)  # 加1避免除零
        except:
            # 如果节点不可达，赋予一个低分值
            centrality_scores['Minclass'][node] = 0.01
    
    return centrality_scores

def identify_key_classes(G, threshold_count=4, top_percent=0.15):
    """
    使用8种方法识别关键类
    :param G: NetworkX 图
    :param threshold_count: 认定为关键类所需的最小投票次数
    :param top_percent: 每种方法选择的比例
    :return: 关键类列表，投票统计
    """
    # 1. 计算所有中心性得分
    centrality_scores = compute_centrality_scores(G)
    
    # 2. 为每种方法选择前15%的节点
    candidate_key_classes = {method: [] for method in centrality_scores}
    all_candidates = []
    
    for method, scores in centrality_scores.items():
        # 按得分降序排序
        sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 计算要选择的节点数量
        num_nodes = len(sorted_nodes)
        num_select = max(1, int(num_nodes * top_percent))  # 至少选择1个节点
        
        # 选择前top_percent的节点
        selected_nodes = [node for node, score in sorted_nodes[:num_select]]
        candidate_key_classes[method] = selected_nodes
        all_candidates.extend(selected_nodes)
    
    # 3. 统计每个节点被选中的次数
    candidate_counter = Counter(all_candidates)
    
    # 4. 确定最终关键类(被至少threshold_count种方法选中)
    key_classes = [node for node, count in candidate_counter.items() if count > threshold_count]
    
    return key_classes, candidate_key_classes, candidate_counter

def print_key_class_results(G, key_classes, candidate_key_classes, candidate_counter, node_mapping=None, log_file="log.txt"):
    """
    打印关键类识别结果，并将关键类ID和类名写入日志文件
    :param G: NetworkX图
    :param key_classes: 关键类节点ID列表
    :param candidate_key_classes: 每种方法的候选类
    :param candidate_counter: 节点被选中次数统计
    :param node_mapping: 节点ID到类名的映射字典
    :param log_file: 日志文件路径
    """
    # 打开日志文件
    with open(log_file, "w") as log:
        # 写入日志标题
        log.write("=" * 80 + "\n")
        log.write("关键类识别结果\n")
        log.write("=" * 80 + "\n\n")
        
        # 1. 显示每种方法选择的候选类
        log.write("每种方法选择的前15%候选类:\n")
        log.write("-" * 60 + "\n")
        for method, candidates in candidate_key_classes.items():
            log.write(f"{method:>15}: {len(candidates)}个节点\n")
        log.write("\n")
        
        # 2. 显示投票统计
        log.write("节点投票统计:\n")
        log.write("-" * 60 + "\n")
        log.write("节点ID\t类名\t被选中次数\t方法列表\n")
        log.write("-" * 60 + "\n")
        
        # 按被选中次数降序排序
        sorted_counter = candidate_counter.most_common()
        
        for node, count in sorted_counter:
            # 获取类名
            class_name = node_mapping[node] if node_mapping and node in node_mapping else f"Node-{node}"
            
            # 获取选中该节点的方法
            methods = [method for method, candidates in candidate_key_classes.items() if node in candidates]
            
            log.write(f"{node}\t{class_name}\t{count}\t\t{', '.join(methods)}\n")
        log.write("\n")
        
        # 3. 显示最终关键类
        log.write(f"最终关键类(被4种以上方法选中):\n")
        log.write("-" * 60 + "\n")
        for node in key_classes:
            # 获取类名
            class_name = node_mapping[node] if node_mapping and node in node_mapping else f"Node-{node}"
            
            # 获取节点度信息
            in_degree = G.in_degree(node, weight='weight')
            out_degree = G.out_degree(node, weight='weight')
            total_degree = in_degree + out_degree
            
            # 获取节点被哪些方法选中
            methods = [method for method, candidates in candidate_key_classes.items() if node in candidates]
            
            log.write(f"节点 {node}: {class_name}\n")
            log.write(f"  度={total_degree} (入度={in_degree}, 出度={out_degree})\n")
            log.write(f"  被{len(methods)}种方法选中: {', '.join(methods)}\n")
            log.write("\n")
        
        # 4. 统计信息
        num_nodes = len(G.nodes())
        num_key_classes = len(key_classes)
        log.write("\n统计摘要:\n")
        log.write("-" * 60 + "\n")
        log.write(f"总节点数: {num_nodes}\n")
        log.write(f"候选关键类总数: {len(candidate_counter)}\n")
        log.write(f"最终关键类数量: {num_key_classes} ({num_key_classes/num_nodes:.2%})\n")
        
        # 5. 只包含关键类ID和类名的简洁列表
        log.write("\n关键类列表 (ID 和 类名):\n")
        log.write("-" * 60 + "\n")
        for node in key_classes:
            class_name = node_mapping[node] if node_mapping and node in node_mapping else f"Node-{node}"
            log.write(f"{node} - {class_name}\n")
    
    # 同时打印一些关键信息到控制台
    print(f"结果已写入日志文件: {log_file}")
    print(f"识别出 {len(key_classes)} 个关键类")
    print(f"详细结果请查看: {log_file}")
    
    return key_classes

# -----------------------------------
# 2. 模型构建
# -----------------------------------

# 2.1 节点评分网络（ScoringNet）
class ScoringNet(nn.Module):
    """
    改进的全连接神经网络，用于将节点嵌入向量映射为初始分值。
    包含更多层、Dropout和BatchNorm以增强泛化能力。
    """

    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.3, use_batch_norm=True):
        super(ScoringNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建多层神经网络
        for i, hidden_dim in enumerate(hidden_dims):
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # 批归一化层
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            # 激活函数
            layers.append(nn.LeakyReLU(0.2))
            # Dropout层
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 最后的输出层
        layers.append(nn.Linear(prev_dim, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 2.2 邻域节点分值聚合 & 2.3 中心性调整
# 模型
class GKCIModel(nn.Module):
    """
    改进的GKCI模型，包含多层邻居分值聚合和中心性调整。
    增加了正则化技术和更灵活的架构，提高泛化能力。
    """

    def __init__(self, G, embedding_dim, hidden_dims=[64, 32], num_layers=2, 
                 num_sampled_neighbors=None, dropout_rate=0.3, use_batch_norm=True):
        super(GKCIModel, self).__init__()
        self.G = G
        self.nodes = list(G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(self.nodes)}
        self.num_nodes = len(self.nodes)
        self.num_layers = num_layers
        self.num_sampled_neighbors = num_sampled_neighbors
        self.dropout_rate = dropout_rate
        
        # 2.1 节点评分网络 - 使用改进的ScoringNet
        self.scoring_net = ScoringNet(
            embedding_dim, 
            hidden_dims=hidden_dims, 
            dropout_rate=dropout_rate, 
            use_batch_norm=use_batch_norm
        )
        
        # region 中心性调整中的参数（beta、gamma)
        if ramdom_gama_beta:
            # 使用更稳定的初始化分布
            self.gamma = nn.Parameter(torch.randn(1) * 0.1 + 3.0)  # 均值为3.0，标准差为0.1
            self.beta = nn.Parameter(torch.randn(1) * 0.1 - 7.0)   # 均值为-7.0，标准差为0.1
        else:
            # 使用之前训练好的较优参数，但增加可调整性
            self.gamma = nn.Parameter(torch.tensor(GKCI_gama))
            self.beta = nn.Parameter(torch.tensor(GKCI_beta))
        logging.info(f"gamma: {self.gamma.item():.4f}, beta: {self.beta.item():.4f}")
        # endregion

        # 新增：自适应中心性缩放机制 - 提高模型在不同图结构上的适应性
        self.centrality_scaling = nn.Parameter(torch.tensor(1.0))
        # 预处理邻接信息
        self.preprocess_adjacency()
        # 特征混合权重 - 使用Sigmoid来保证范围在0-1之间
        self.mixing_weight_raw = nn.Parameter(torch.tensor(0.0))  # 初始化为0，经过sigmoid后为0.5   
        # 新增：噪声抑制阈值
        self.noise_threshold = nn.Parameter(torch.tensor(0.05))

        # 多层更新函数，每层有独立的全连接层，增加了Dropout和BatchNorm
        self.update_fc_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = []
            # 线性层 - 增加宽度以提高表达能力
            layer.append(nn.Linear(3, 24))
            # 批归一化
            if use_batch_norm:
                layer.append(nn.BatchNorm1d(24))
            # 使用LeakyReLU代替ReLU提高梯度流动性
            layer.append(nn.LeakyReLU(0.1))
            # Dropout
            layer.append(nn.Dropout(dropout_rate))
            # 隐藏层
            layer.append(nn.Linear(24, 12))
            layer.append(nn.LeakyReLU(0.1))
            layer.append(nn.Dropout(dropout_rate * 0.5))  # 减少后层的Dropout以保留更多信息
            # 输出层
            layer.append(nn.Linear(12, 1))
            layer.append(nn.Sigmoid())
            
            self.update_fc_layers.append(nn.Sequential(*layer))

    # 前继邻居和后继邻居的索引及对应的权重
    def preprocess_adjacency(self):
        self.pred_indices = []
        self.pred_weights = []
        self.succ_indices = []
        self.succ_weights = []

        for node in self.nodes:
            # 前继邻居
            preds = list(self.G.predecessors(node))
            pred_idx = [self.node_to_idx[n] for n in preds]
            pred_w = [self.G[n][node]['weight'] for n in preds]
            self.pred_indices.append(pred_idx)
            self.pred_weights.append(pred_w)

            # 后继邻居
            succs = list(self.G.successors(node))
            succ_idx = [self.node_to_idx[n] for n in succs]
            succ_w = [self.G[node][n]['weight'] for n in succs]
            self.succ_indices.append(succ_idx)
            self.succ_weights.append(succ_w)


    def forward(self, embeddings, training=True):
        device = embeddings.device
        # 初始评分
        layer_scores = self.scoring_net(embeddings).view(-1)  # [num_nodes]
        
        # 保存原始节点分数以供后续使用
        original_scores = layer_scores.clone()

        for layer in range(self.num_layers):
            aggregated_scores_in = torch.zeros(self.num_nodes, device=device)
            aggregated_scores_out = torch.zeros(self.num_nodes, device=device)

            # region 每个节点前继和后继邻居的分数聚合（aggregated_scores_in, aggregated_scores_out)
            for idx in range(self.num_nodes):
                # 前继邻居
                pred_idx = self.pred_indices[idx]
                pred_w = self.pred_weights[idx]

                if len(pred_idx) > 0:
                    # 如果需要采样邻居
                    if training and self.num_sampled_neighbors and len(pred_idx) > self.num_sampled_neighbors:
                        sampled = self.num_sampled_neighbors
                        # 随机采样邻居
                        sampled_indices = random.sample(range(len(pred_idx)), sampled)
                        pred_idx_sample = [pred_idx[i] for i in sampled_indices]
                        pred_w_sample = [pred_w[i] for i in sampled_indices]
                    else:
                        pred_idx_sample = pred_idx
                        pred_w_sample = pred_w
                    
                    pred_idx_tensor = torch.tensor(pred_idx_sample, dtype=torch.long, device=device)
                    pred_w_tensor = torch.tensor(pred_w_sample, dtype=torch.float32, device=device)
                    
                    # 加权聚合前继邻居的分数
                    s_in = (pred_w_tensor * layer_scores[pred_idx_tensor]).sum()
                    
                    # 应用自注意力机制
                    if len(pred_idx_sample) > 1:
                        attention_scores = F.softmax(layer_scores[pred_idx_tensor], dim=0)
                        s_in = (attention_scores * pred_w_tensor * layer_scores[pred_idx_tensor]).sum()
                    else:
                        s_in = (pred_w_tensor * layer_scores[pred_idx_tensor]).sum()
                else:
                    s_in = torch.tensor(0.0, device=device)

                # 后继邻居
                succ_idx = self.succ_indices[idx]
                succ_w = self.succ_weights[idx]

                if len(succ_idx) > 0:
                    if training and self.num_sampled_neighbors and len(succ_idx) > self.num_sampled_neighbors:
                        sampled = self.num_sampled_neighbors
                        sampled_indices = random.sample(range(len(succ_idx)), sampled)
                        succ_idx_sample = [succ_idx[i] for i in sampled_indices]
                        succ_w_sample = [succ_w[i] for i in sampled_indices]
                    else:
                        succ_idx_sample = succ_idx
                        succ_w_sample = succ_w
                    
                    succ_idx_tensor = torch.tensor(succ_idx_sample, dtype=torch.long, device=device)
                    succ_w_tensor = torch.tensor(succ_w_sample, dtype=torch.float32, device=device)
                    
                    # 加权聚合邻居的分数
                    if len(succ_idx_sample) > 1:
                        attention_scores = F.softmax(layer_scores[succ_idx_tensor], dim=0)
                        s_out = (attention_scores * succ_w_tensor * layer_scores[succ_idx_tensor]).sum()
                    else:
                        s_out = (succ_w_tensor * layer_scores[succ_idx_tensor]).sum()
                else:
                    s_out = torch.tensor(0.0, device=device)

                aggregated_scores_in[idx] = s_in
                aggregated_scores_out[idx] = s_out
            # endregion
            
            # 层评分更新--聚合后激活
            combined = torch.stack([layer_scores, aggregated_scores_in, aggregated_scores_out], dim=1)  # [num_nodes, 3]
            layer_scores = self.update_fc_layers[layer](combined).squeeze()  # [num_nodes]
            
            # 改进的残差连接 - 使用自适应残差系数
            layer_idx_factor = (layer + 1) / self.num_layers  # 基于层索引的系数
            residual_alpha = 0.7 + 0.2 * layer_idx_factor  # 从0.7逐渐增加到0.9
            layer_scores = residual_alpha * layer_scores + (1 - residual_alpha) * original_scores

        # 中心性调整  （论文中中心性调整只用入度）
        degrees = torch.tensor(
            # [self.G.in_degree(node) + self.G.out_degree(node) for node in self.nodes],
            [self.G.in_degree(node) for node in self.nodes],
            dtype=torch.float32,
            device=device
        )
        epsilon = 1e-6

        # region 他自己改进的中心性调整
        # 改进的中心性计算 - 使用自适应缩放
        degrees_scaled = degrees * self.centrality_scaling
        c_v = torch.log(degrees_scaled + epsilon)
        c_star_v = self.gamma * c_v + self.beta  # [num_nodes]

        
        # 使用Tanh对极值进行平滑，再通过线性变换将范围调整到合适区间
        c_star_v_smooth = 5.0 * torch.tanh(c_star_v / 5.0)
        
        # 使用平滑的调整系数
        final_scores = torch.sigmoid(c_star_v_smooth * layer_scores)

        # 动态混合权重 - 使用sigmoid确保在0-1之间
        mixing_weight = torch.sigmoid(self.mixing_weight_raw)
        
        # 使用动态混合权重 混合新旧分数
        mixed_scores = mixing_weight * final_scores + (1 - mixing_weight) * torch.sigmoid(original_scores)
        
        # 噪声抑制 - 如果分数非常低，可能是噪声，将其进一步降低
        noise_threshold = torch.sigmoid(self.noise_threshold)  # 确保阈值在0-1之间
        noise_mask = (mixed_scores < noise_threshold).float()
        mixed_scores = mixed_scores * (1.0 - noise_mask * 0.5)  # 对低于阈值的分数乘以0.5
        
        # 确保输出值严格在0到1之间（应用截断）
        mixed_scores = torch.clamp(mixed_scores, 0.0, 1.0)
        # endregion
        
        # 用论文的中心性调整要注释掉上面region；改return为final_scores
        # region 论文中中心性调整得到final_scores
        # c_v = torch.log(degrees + epsilon)  
        # c_star_v = self.gamma * c_v + self.beta  
        # final_scores = torch.sigmoid(c_star_v * layer_scores)  
        # endregion
        
        return mixed_scores


# region 2.4 模型训练
def train_model(G, embeddings, labels, num_epochs=1000, learning_rate=0.0001, model=None, prev_model_path=None, validation_data=None):
    """
    训练 GKCI 模型，增强版本支持更多迁移学习功能。
    """
    start_epoch = 0
    if model is None:
        embedding_dim = embeddings.shape[1]
        hidden_dims = preset_best_param.get('hidden_dims', [64, 32])
        model = GKCIModel(
            G, 
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            num_layers=preset_best_param['num_layers'],
            num_sampled_neighbors=preset_best_param['num_sampled_neighbors'],
            dropout_rate=preset_best_param.get('dropout_rate', 0.3),
            use_batch_norm=preset_best_param.get('batch_norm', True)
        )
    
    # region 有已训练的模型(迁移学习)
    prev_trained_epochs = 0
    if prev_model_path and os.path.exists(prev_model_path):
        try:
            checkpoint = torch.load(prev_model_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])       # 将参数加载到模型中
            prev_trained_epochs = checkpoint.get('epoch', 0)
            logging.info(f"成功加载上一次训练的模型，之前已训练 {prev_trained_epochs} 轮")
            
            # 恢复固定参数gamma和beta
            if 'gamma' in checkpoint and 'beta' in checkpoint and not ramdom_gama_beta:
                with torch.no_grad():
                    model.gamma.copy_(torch.tensor(checkpoint['gamma']))
                    model.beta.copy_(torch.tensor(checkpoint['beta']))
                logging.info(f"从上一次训练加载gamma={model.gamma.item():.4f}, beta={model.beta.item():.4f}")
                
                # 恢复动态参数
                if hasattr(model, 'centrality_scaling') and 'centrality_scaling' in checkpoint:
                    model.centrality_scaling.copy_(torch.tensor(checkpoint['centrality_scaling']))
                    logging.info(f"从上一次训练加载centrality_scaling={model.centrality_scaling.item():.4f}")
                
                if hasattr(model, 'mixing_weight_raw') and 'mixing_weight' in checkpoint:
                    # 反算raw值
                    mixing_weight = checkpoint['mixing_weight']
                    raw_value = torch.log(torch.tensor(mixing_weight) / (1 - torch.tensor(mixing_weight) + 1e-8))
                    model.mixing_weight_raw.copy_(raw_value)
                    logging.info(f"从上一次训练加载mixing_weight={mixing_weight:.4f}")
                
                if hasattr(model, 'noise_threshold') and 'noise_threshold' in checkpoint:
                    # 反算raw值
                    noise_threshold = checkpoint['noise_threshold']
                    raw_value = torch.log(torch.tensor(noise_threshold) / (1 - torch.tensor(noise_threshold) + 1e-8))
                    model.noise_threshold.copy_(raw_value)
                    logging.info(f"从上一次训练加载noise_threshold={noise_threshold:.4f}")
        except Exception as e:
            logging.error(f"加载模型失败: {str(e)}")
            logging.info("将创建新模型进行训练")
    # endregion
    
    # 设置特征增强参数
    use_feature_augmentation = True  # 是否使用特征增强
    augmentation_level = 0.05        # 初始扰动水平
    augmentation_decay = 0.995       # 每个epoch衰减因子
    
    # region 自适应学习率策略（迁移学习）
    if transfer_learning_settings['adaptive_learning_rate']:
        # 迁移学习--降低学习率
        if enable_transfer_learning and (prev_model_path or model is not None):
            initial_lr = learning_rate * 0.3
            logging.info(f"迁移学习: 降低初始学习率至 {initial_lr:.6f}")
        else:
            initial_lr = learning_rate
            
        # 迁移学习--优化器AdamW
        weight_decay = preset_best_param.get('weight_decay', 0.01)  # 增加权重衰减提高泛化
        optimizer = optim.AdamW(
            # model.parameters(), 
            [p for p in model.parameters() if p.requires_grad],
            lr=initial_lr, 
            weight_decay=weight_decay,# 权重衰减
            betas=(0.9, 0.999),  # 标准设置
            eps=1e-8,            # 数值稳定性
            amsgrad=True         # 使用AMSGrad变种改进收敛
        )
        
        # 迁移学习--学习率调度OneCycleLR
        if (num_epochs > 0):
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=initial_lr * 3,              # 峰值学习率
                total_steps=num_epochs,
                pct_start=0.3,                     # 前30%时间提升学习率，后70%下降
                div_factor=10.0,                   # 初始学习率 = max_lr/10
                final_div_factor=100.0             # 最终学习率 = max_lr/1000
            )
        else:
            scheduler = None
    else:
        # 原始方法 权重衰减、固定学习率
        weight_decay = preset_best_param.get('weight_decay', 0.001)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # endregion
    
    # region 已有优化器状态 加载
    # if prev_model_path and os.path.exists(prev_model_path):
    #     try:
    #         checkpoint = torch.load(prev_model_path)
    #         if 'optimizer_state_dict' in checkpoint:
    #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #             logging.info("成功加载优化器状态")
    #     except Exception as e:
    #         logging.error(f"加载优化器状态失败: {str(e)}")
    if prev_model_path and os.path.exists(prev_model_path):
        try:
            checkpoint = torch.load(prev_model_path)
            if 'optimizer_state_dict' in checkpoint:
                # 检查参数形状是否一致
                model_state = model.state_dict()
                optimizer_state = checkpoint['optimizer_state_dict']
                
                mismatch = False
                for p, (k, v) in zip(model.parameters(), optimizer_state['state'].items()):
                    if p.shape != v['exp_avg'].shape:
                        logging.warning(f"参数形状不匹配: {k} (当前{p.shape} vs 保存的{v['exp_avg'].shape})")
                        mismatch = True
                
                if not mismatch:
                    optimizer.load_state_dict(optimizer_state)
                    logging.info("优化器状态加载成功")
                else:
                    logging.warning("优化器状态不匹配，将重新初始化")
        except Exception as e:
            logging.error(f"加载优化器状态失败: {str(e)}")
    # endregion

    # 平衡的损失函数BCEloss - 处理类别不平衡问题
    criterion = nn.BCELoss()
    pos_weight = torch.tensor([(labels == 0).sum().item()/max(1, (labels == 1).sum().item())])# 正样本权重
    criterion_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)        # 正样本损失加权
    
    # 渐进式解冻（迁移学习）
    if transfer_learning_settings['gradual_unfreezing'] and enable_transfer_learning:
        # 解冻点
        unfreeze_epochs = [num_epochs // 10, num_epochs // 5, num_epochs // 3]
        logging.info(f"设置渐进式解冻点: {unfreeze_epochs}")

    # 将嵌入和标签转换为张量
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # device设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    embeddings_tensor = embeddings_tensor.to(device)
    labels_tensor = labels_tensor.to(device)
    if transfer_learning_settings['adaptive_learning_rate']:
        pos_weight = pos_weight.to(device)
    
    # region 验证集设置
    if validation_data:
        val_embeddings, val_labels = validation_data
        val_embeddings_tensor = torch.tensor(val_embeddings, dtype=torch.float32).to(device)
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).to(device)
        
        # 确保模型目录存在
        os.makedirs(model_dir, exist_ok=True)
        
        # 设置早停机制
        early_stopping = EarlyStopping(
            patience=early_stopping_patience, 
            delta=early_stopping_delta,
            path=os.path.join(model_dir, "best_model.pth"),
            verbose=True
        )
    # endregion
    
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'epochs': [],
        'learning_rates': [],
        'augmentation_levels': []  # 新增：记录增强水平
    }
    
    # 计算实际的训练轮数
    total_epochs = prev_trained_epochs + num_epochs     # 加上模型已经训练的轮数
    logging.info(f"将训练 {num_epochs} 轮，训练后总轮数将达到 {total_epochs}")

    # 类权重计算
    pos_counts = torch.sum(labels_tensor).item()    # 正样本数量
    neg_counts = len(labels_tensor) - pos_counts    # 负样本数量
    pos_weight = torch.tensor([neg_counts / max(1, pos_counts)]).to(device)     # 正样本权重
    class_weights = torch.ones_like(labels_tensor).to(device)
    class_weights[labels_tensor == 1] = pos_weight  # 正样本权重赋值

    for epoch in range(prev_trained_epochs, total_epochs): #就是num_epochs的轮数
        # 训练模式
        model.train()
        
        # 迁移学习-渐进式解冻 - 随着训练的进行逐渐解冻更多层
        if transfer_learning_settings['gradual_unfreezing'] and enable_transfer_learning:
            relative_epoch = epoch - prev_trained_epochs
            
            # 检查当前轮数是否在解冻点
            if relative_epoch in unfreeze_epochs:
                if relative_epoch == unfreeze_epochs[0]:
                    # 解锁scoring_net的后半部分
                    for layer in list(model.scoring_net.modules())[-(len(list(model.scoring_net.modules()))//2):]:
                        if isinstance(layer, nn.Linear):
                            for param in layer.parameters():
                                param.requires_grad = True
                    logging.info("解冻scoring_net的后半部分")
                
                elif relative_epoch == unfreeze_epochs[1]:
                    # 解锁所有scoring_net层
                    for param in model.scoring_net.parameters():
                        param.requires_grad = True
                    logging.info("解冻所有scoring_net层")
                
                elif relative_epoch == unfreeze_epochs[2]:
                    # 解锁所有参数
                    for param in model.parameters():
                        param.requires_grad = True
                    logging.info("解冻所有模型参数")
        
        optimizer.zero_grad()
        
        # 特征增强
        if use_feature_augmentation and epoch < total_epochs * 0.8:  # 只在前80%的训练过程中使用
            current_augmentation = augmentation_level * (augmentation_decay ** epoch)
            history['augmentation_levels'].append(current_augmentation)
            aug_features = augment_features(embeddings_tensor, current_augmentation)
            outputs = model(aug_features, training=True)
        else:
            outputs = model(embeddings_tensor, training=True)

        # 确保outputs结果在有效范围内
        # region ===为什么要截断而不Sigmoid
        # outputs = torch.sigmoid(outputs)

        min_train = outputs.min().item()
        max_train = outputs.max().item()
        if min_train < 0.0 or max_train > 1.0:
            logging.warning(f"警告：训练outputs的值超出范围[0,1]! min={min_train}, max={max_train}")
        outputs = torch.clamp(outputs, 0.0, 1.0)    # outputs模型预测为正的结果
        # endregion
        
        # region损失计算
        # 样本不平衡时（正样本数少于总样本的30%）使用动态加权的损失函数
        if (labels == 1).sum().item() < len(labels) * 0.3:
            logging.info("检测到样本不平衡，使用动态加权的损失函数")
            # pt越小，分错的多；pt越大，分对的多
            pt = outputs * labels_tensor + (1 - outputs) * (1 - labels_tensor)
            #分错越多，权重越大
            focal_weight = (1 - pt) ** 2  
            # 结合类别权重和focal权重
            combined_weight = class_weights * focal_weight
            
            # 二元交叉熵损失，结合和类别权重和focal权重
            bce_loss = F.binary_cross_entropy(outputs, labels_tensor, reduction='none')
            loss = (bce_loss * combined_weight).mean()
        else:
            logging.info("样本平衡，使用标准BCE损失函数")
            # 标准BCE损失
            loss = criterion(outputs, labels_tensor)
        # endregion
        
        loss.backward()
        optimizer.step()
        
        # 学习率更新
        if transfer_learning_settings['adaptive_learning_rate']:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            history['learning_rates'].append(current_lr)
        
        # 计算准确率accuracy
        with torch.no_grad():
            preds = (outputs > 0.5).float()         # >0.5 视为正类，≤0.5 视为负类
            correct = (preds == labels_tensor).sum().item()     # 预测与真实标签匹配的样本数
            accuracy = correct / len(labels_tensor)

        # 如果有验证集，计算验证集上的性能
        if validation_data:
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_embeddings_tensor, training=False)  #验证集结果
                
                # region ===为什么要截断而不Sigmoid
                # val_outputs = torch.sigmoid(val_outputs)  # 如果模型输出是logits，则需要Sigmoid

                # 确保验证输出在有效范围内
                val_outputs = torch.clamp(val_outputs, 0.0, 1.0)
                # 检查值是否真的在0到1之间
                min_val = val_outputs.min().item()
                max_val = val_outputs.max().item()
                if min_val < 0.0 or max_val > 1.0:
                    logging.warning(f"警告：val_outputs的值超出范围[0,1]! min={min_val}, max={max_val}")
                    # 再次确保在0-1范围内
                    val_outputs = torch.clamp(val_outputs, 0.0, 1.0)
                # endregion
                
                val_loss = criterion(val_outputs, val_labels_tensor)        # 验证集损失
                val_preds = (val_outputs > 0.5).float()                     # 二值化预测结果
                val_correct = (val_preds == val_labels_tensor).sum().item() # 验证集正确样本数
                val_accuracy = val_correct / len(val_labels_tensor)         # 验证集准确率
                
                # 早停检查
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    logging.info(f"早停触发，在 {epoch + 1} 轮停止训练")
                    # 加载最佳模型
                    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))
                    break

        # 每output_frequency轮输出一次训练状态
        if (epoch + 1) % output_frequency == 0:
            if validation_data:
                lr_info = f", lr: {current_lr:.6f}" if transfer_learning_settings['adaptive_learning_rate'] else ""
                aug_info = f", aug: {current_augmentation:.6f}" if use_feature_augmentation and epoch < total_epochs * 0.8 else ""
                logging.info(
                    f'epoch [{epoch + 1}/{total_epochs}], loss: {loss.item():.4f}, accuracy: {accuracy * 100:.2f}%, '
                    f'val_loss: {val_loss.item():.4f}, val_accuracy: {val_accuracy * 100:.2f}%, '
                    f'gamma: {model.gamma.item():.4f}, beta: {model.beta.item():.4f}, '
                    f'centrality_scaling: {model.centrality_scaling.item():.4f}{lr_info}{aug_info}'
                )
                history['val_loss'].append(val_loss.item())
                history['val_accuracy'].append(val_accuracy)
            else:
                lr_info = f", lr: {current_lr:.6f}" if transfer_learning_settings['adaptive_learning_rate'] else ""
                logging.info(
                    f'epoch [{epoch + 1}/{total_epochs}], loss: {loss.item():.4f}, accuracy: {accuracy * 100:.2f}%, '
                    f'gamma: {model.gamma.item():.4f}, beta: {model.beta.item():.4f}, '
                    f'centrality_scaling: {model.centrality_scaling.item():.4f}{lr_info}'
                )
            
            # 记录历史数据
            history['loss'].append(loss.item())
            history['accuracy'].append(accuracy)
            history['epochs'].append(epoch + 1)

    return model, history, optimizer, total_epochs
# endregion

if create_hybrid:
    combined_net_file = create_combined_dataset()

if __name__ == "__main__":
    # 创建示例图 (在实际应用中替换为您的图)
    G, node_mapping = generate_sample_graph()
    
    # 识别关键类
    key_classes, candidate_key_classes, candidate_counter = identify_key_classes(G)
    
    # 打印结果
    print_key_class_results(G, key_classes, candidate_key_classes, candidate_counter, node_mapping)

    # 计算节点嵌入  得到节点向量(embeddings)
    embeddings = compute_node_embeddings(G, dimensions=Node2Vec_dimensions)
    
    # 使用额外的中心性指标增强特征
    if use_multiple_centrality:
        centrality_features = compute_centrality_features(G)
        combined_features = combine_features(embeddings, centrality_features)
        logging.info(f"使用组合特征: 嵌入维度={embeddings.shape[1]}, 中心性特征维度={centrality_features.shape[1]}, 总维度={combined_features.shape[1]}")
    else:
        combined_features = combine_features(embeddings)
        logging.info(f"仅使用嵌入特征: 维度={combined_features.shape[1]}")

    
