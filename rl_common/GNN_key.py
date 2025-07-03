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


# 是否使用真实的数据
is_real_data = True
# 数据集名称
dataset = "daisy"
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
use_predefined_key_classes = True
# 找不到精确匹配时是否允许模糊匹配
allow_fuzzy_matching = True
# 没有找到的关键类是否通过中心性计算来补充
auto_complete_missing_classes = True
# 是否使用上次训练的模型继续训练
use_previous_model = False
# 是否启用迁移学习模式
enable_transfer_learning = False
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
num_epochs = 1000  # 训练轮数

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
result_dir = "/home/ps/jy_exp/output/GNN_res"
os.makedirs(result_dir, exist_ok=True)

# 历史模型保存路径
historical_models_dir = "/home/ps/jy_exp/output/GNN_res/historicalModels"
os.makedirs(historical_models_dir, exist_ok=True)

# 获取下一个结果文件的序号，命名为 result-1.txt, result-2.txt, ...
def get_next_result_file():
    log_dir = os.path.join(result_dir, "logs")
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


# 最近的训练模型--（模型路径, 版本号）
# 根据最新的版本号加载模型路径
def load_latest_model():
    existing_dirs = [d for d in os.listdir(historical_models_dir) if os.path.isdir(os.path.join(historical_models_dir, d))]
    pattern = re.compile(r'Train-(\d+)')
    max_num = 0
    latest_dir = None
    
    for dirname in existing_dirs:
        match = pattern.match(dirname)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                latest_dir = dirname
    
    if latest_dir:
        model_path = os.path.join(historical_models_dir, latest_dir, "final_model.pth")
        if os.path.exists(model_path):
            logging.info(f"加载最近训练的模型: {model_path}")
            return model_path, max_num
    
    logging.info("未找到之前训练的模型，将创建新模型")
    return None, None

# 在save_model函数前添加数据增强函数
def augment_features(features, augmentation_level=0.1):
    """
    对输入特征进行数据增强，提高模型泛化能力。
    
    :param features: 输入特征张量
    :param augmentation_level: 增强强度
    :return: 增强后的特征张量
    """
    if not isinstance(features, torch.Tensor):
        features = torch.tensor(features, dtype=torch.float32)
        
    # 特征缩放扰动
    scaling_noise = torch.randn_like(features) * augmentation_level + 1.0
    features_scaled = features * scaling_noise
    
    # 特征置换扰动 - 随机置换部分特征维度
    batch_size, feature_dim = features.shape
    if feature_dim > 10:  # 只有在特征维度较大时才执行
        num_dims_to_permute = max(1, int(feature_dim * 0.05))  # 置换5%的特征维度
        dims_to_permute = torch.randperm(feature_dim)[:num_dims_to_permute]
        
        # 创建随机置换索引
        perm_indices = torch.randperm(batch_size)
        
        # 只置换选定的维度
        features_perm = features_scaled.clone()
        for dim in dims_to_permute:
            features_perm[:, dim] = features_scaled[perm_indices, dim]
        
        return features_perm
    
    return features_scaled

# 保存模型信息到..GKCI/GNN/GKCI/historicalModels/Train-{train_id}目录下
def save_model(model, optimizer, epoch, train_id, split_idx=None):
    model_dir = os.path.join(historical_models_dir, f"Train-{train_id}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 如果是多次划分的训练，保存不同的模型文件
    if split_idx is not None:
        model_path = os.path.join(model_dir, f"split_{split_idx}_model.pth")
    else:
        model_path = os.path.join(model_dir, "final_model.pth")
    
    state = {
        'model_state_dict': model.state_dict(),                     # 模型参数
        'optimizer_state_dict': optimizer.state_dict(),             # 优化器参数
        'epoch': epoch,                                             # 当前训练轮数   
        'gamma': model.gamma.item(),                                # 中心性调整参数
        'beta': model.beta.item(),                                  # 中心性偏置参数    
        'embedding_dim': model.scoring_net.layers[0].in_features    # 嵌入维度
    }
    
    # 保存动态参数到模型state中
    if hasattr(model, 'centrality_scaling'):
        state['centrality_scaling'] = model.centrality_scaling.item()
    
    if hasattr(model, 'mixing_weight_raw'):
        state['mixing_weight'] = torch.sigmoid(model.mixing_weight_raw).item()
    
    if hasattr(model, 'noise_threshold'):
        state['noise_threshold'] = torch.sigmoid(model.noise_threshold).item()
    
    torch.save(state, model_path)
    logging.info(f"模型已保存到: {model_path}")
    
    # 保存模型配置信息
    config_path = os.path.join(model_dir, "model_config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Network Type: {net_type}\n")
        f.write(f"Embedding Dimensions: {Node2Vec_dimensions}\n")
        f.write(f"Learning Rate: {preset_best_param['learning_rate']}\n")
        f.write(f"Num Sampled Neighbors: {preset_best_param['num_sampled_neighbors']}\n")
        f.write(f"Num Layers: {preset_best_param['num_layers']}\n")
        f.write(f"Final Gamma: {model.gamma.item()}\n")
        f.write(f"Final Beta: {model.beta.item()}\n")
        
        # 保存动态参数到文件中
        if hasattr(model, 'centrality_scaling'):
            f.write(f"Centrality Scaling: {model.centrality_scaling.item()}\n")
        
        if hasattr(model, 'mixing_weight_raw'):
            f.write(f"Mixing Weight: {torch.sigmoid(model.mixing_weight_raw).item()}\n")
        
        if hasattr(model, 'noise_threshold'):
            f.write(f"Noise Threshold: {torch.sigmoid(model.noise_threshold).item()}\n")
            
        f.write(f"Epochs: {epoch}\n")
    
    return model_path

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
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=initial_lr * 3,              # 峰值学习率
            total_steps=num_epochs,
            pct_start=0.3,                     # 前30%时间提升学习率，后70%下降
            div_factor=10.0,                   # 初始学习率 = max_lr/10
            final_div_factor=100.0             # 最终学习率 = max_lr/1000
        )
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

# -----------------------------------
# 3. 关键类识别
# -----------------------------------

# 不同k值下的召回率和精确率（top-k）
def compute_recall_precision_at_k(labels, scores, k_list):
    """
    计算不同 K 值下的召回率和精确率。

    :param labels: 真实标签数组
    :param scores: 模型预测分数数组
    :param k_list: K 值列表
    :return: 召回率列表，精确率列表
    """
    sorted_indices = np.argsort(scores)[::-1]  # 分数从高到低排序
    total_positives = np.sum(labels)
    recalls = []
    precisions = []
    for k in k_list:
        top_k_indices = sorted_indices[:k]
        top_k_labels = labels[top_k_indices]
        true_positives = np.sum(top_k_labels)
        recall = true_positives / total_positives if total_positives > 0 else 0
        precision = true_positives / k
        recalls.append(recall)
        precisions.append(precision)
    return recalls, precisions


# 3.2 评价指标
# 进行参数调优，searching_best_param = True时调用
def parameter_tuning(G, labels, embeddings):
    """
    进行参数调优，寻找最佳的学习率、采样邻居数量和层数组合。

    :param G: NetworkX 图
    :param labels: 节点标签数组
    :param embeddings: 节点嵌入向量数组
    :return: 最佳参数字典
    """
    # 定义参数搜索空间
    learning_rates = [0.01, 0.001, 0.0001]         # 学习率
    num_sampled_neighbors_list = [5, 10]    # 采样邻居数量
    num_layers_list = [1, 2]                # 层数

    best_score = 0
    best_params = {}

    for lr in learning_rates:
        for num_sampled_neighbors in num_sampled_neighbors_list:
            for num_layers in num_layers_list:
                logging.info(
                    f"\nTraining with lr={lr}, num_sampled_neighbors={num_sampled_neighbors}, num_layers={num_layers}")
                # 初始化模型
                model = GKCIModel(
                    G,
                    embedding_dim=embeddings.shape[1],
                    num_layers=num_layers,
                    num_sampled_neighbors=num_sampled_neighbors
                )
                print("参数调优")
                # 训练模型
                model, history, optimizer, _ = train_model(
                    G,
                    embeddings,
                    labels,
                    num_epochs=num_epochs,
                    learning_rate=lr,
                    model=model
                )

                # 评估模型
                model.eval()
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                embeddings_tensor = embeddings_tensor.to(device)
                with torch.no_grad():
                    final_scores = model(embeddings_tensor)
                node_scores = final_scores.cpu().numpy()
                labels_np = labels

                # 计算指标
                k_list = [5, 10, 15]
                recalls, precisions = compute_recall_precision_at_k(labels_np, node_scores, k_list)
                avg_precision = np.mean(precisions)
                logging.info(f'Recall@K: {recalls}')
                logging.info(f'Precision@K: {precisions}')
                logging.info(f'Average Precision: {avg_precision:.4f}')

                # 更新最佳参数
                if avg_precision > best_score:
                    best_score = avg_precision
                    best_params = {
                        'learning_rate': lr,
                        'num_sampled_neighbors': num_sampled_neighbors,
                        'num_layers': num_layers
                    }

    logging.info("\nBest Parameters:")
    logging.info(best_params)
    logging.info(f'Best Average Precision: {best_score:.4f}')
    return best_params


# 可视化类依赖网络（未调用）
def visualize_graph(G, node_scores, labels, title="Graph with Node Importance"):
    """
    可视化类依赖网络，节点颜色和大小根据重要性分数变化。

    :param G: NetworkX 图
    :param node_scores: 节点重要性分数列表
    :param labels: 节点标签数组
    :param title: 图形标题
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # 使用 spring layout 进行节点布局
    pos = nx.spring_layout(G, seed=42)

    # 标准化分数以匹配颜色映射
    norm = mcolors.Normalize(vmin=min(node_scores), vmax=max(node_scores))
    cmap = cm.viridis

    # 根据分数设置节点颜色
    node_colors = [cmap(norm(score)) for score in node_scores]

    # 根据分数设置节点大小
    if max(node_scores) != min(node_scores):
        node_sizes = [
            300 + 700 * (score - min(node_scores)) / (max(node_scores) - min(node_scores))
            for score in node_scores
        ]
    else:
        node_sizes = [500 for _ in node_scores]

    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.9,
        ax=ax
    )

    # 绘制有向边，边宽根据权重调整
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, width=edge_weights,
        alpha=0.5, arrowstyle='->', arrowsize=15, ax=ax
    )

    # 绘制节点标签
    labels_dict = {node: f"{node}" for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels_dict,
        font_size=12, ax=ax
    )

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(node_scores)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Node Importance Score')

    plt.title(title)
    plt.axis('off')
    plt.show()


# region 绘制loss和accuracy趋势图的函数
def plot_training_history(history, title="Trainning Process"):
    """
    绘制训练过程中的loss和accuracy趋势图。

    :param history: 包含训练历史数据的字典
    :param title: 图表标题
    """
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制loss曲线（使用左侧Y轴）
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['epochs'], history['loss'], color=color, linestyle='-', label='Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # 创建共享X轴的第二个Y轴，用于accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(history['epochs'], history['accuracy'], color=color, linestyle='-', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 添加网格线和标题
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.title(title)
    
    # 添加图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # 调整布局并显示
    fig.tight_layout()
    
    # 保存图像
    png_dir = os.path.join(result_dir, "trainning_png")
    os.makedirs(png_dir, exist_ok=True)
    plt.savefig(os.path.join(png_dir, f'{dataset}_training_history.png'))
    plt.show()
# endregion


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


# EarlyStopping类
class EarlyStopping:
    """
    早停机制，防止过拟合，提高模型的泛化能力。
    """
    def __init__(self, patience=7, delta=0, path='checkpoint.pt', verbose=False):
        """
        :param patience: 容忍多少个epoch内验证集性能不提升，默认7
        :param delta: 监控指标最小变化阈值，小于这个阈值认为没有提升，默认0
        :param path: 模型保存路径
        :param verbose: 是否打印早停信息
        """
        self.patience = patience
        self.delta = delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        '''保存模型'''
        if self.verbose:
            logging.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 模型保存目录
model_dir = "/home/ps/jy_exp/output/GNN_res/models"
# 训练好的通用模型路径
general_model_path = os.path.join(model_dir, "general_model.pth")

# 保存通用模型数据（迁移学习）到../models/general_model.pth
def save_general_model(model, optimizer, epoch, metadata=None):
    """
    保存通用模型，用于迁移学习。
    
    :param model: 训练好的模型
    :param optimizer: 优化器
    :param epoch: 当前轮数
    :param metadata: 模型元数据（可选）
    """
    # 确保模型目录存在
    os.makedirs(model_dir, exist_ok=True)
    
    # 准备模型状态
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'gamma': model.gamma.item(),
        'beta': model.beta.item(),
        'embedding_dim': model.scoring_net.layers[0].in_features,  # 保存输入维度
    }
    
    # 保存新增的参数
    if hasattr(model, 'centrality_scaling'):
        state['centrality_scaling'] = model.centrality_scaling.item()
    
    if hasattr(model, 'mixing_weight_raw'):
        state['mixing_weight'] = torch.sigmoid(model.mixing_weight_raw).item()
    
    if hasattr(model, 'noise_threshold'):
        state['noise_threshold'] = torch.sigmoid(model.noise_threshold).item()
    
    # 如果有元数据，添加到状态中
    if metadata:
        state.update(metadata)
    
    # 保存模型
    torch.save(state, general_model_path)
    logging.info(f"通用模型已保存到: {general_model_path}")
    
    # 保存模型配置到文件
    config_path = os.path.join(model_dir, "general_model_config.json")
    with open(config_path, 'w') as f:
        config = {
            'embedding_dim': model.scoring_net.layers[0].in_features,
            'gamma': float(model.gamma.item()),
            'beta': float(model.beta.item()),
            'epoch': epoch,
            'dataset': dataset,
            'net_type': net_type,
            'node2vec_p': node2vec_p_param,
            'node2vec_q': node2vec_q_param,
            'node2vec_dim': Node2Vec_dimensions,
        }
        
        if hasattr(model, 'centrality_scaling'):
            config['centrality_scaling'] = float(model.centrality_scaling.item())
        
        if hasattr(model, 'mixing_weight_raw'):
            config['mixing_weight'] = float(torch.sigmoid(model.mixing_weight_raw).item())
        
        if hasattr(model, 'noise_threshold'):
            config['noise_threshold'] = float(torch.sigmoid(model.noise_threshold).item())
        
        if metadata:
            config.update({k: (float(v) if isinstance(v, torch.Tensor) else v) for k, v in metadata.items()})
        json.dump(config, f, indent=4)


# 加载通用模型（迁移学习）
def load_general_model(G, embedding_dim):
    """
    加载通用模型，用于迁移学习。
    
    :param G: 目标数据集的图
    :param embedding_dim: 嵌入向量的维度
    :return: 加载的模型或None（如果加载失败）
    """
    if not os.path.exists(general_model_path):
        logging.info(f"未找到通用模型: {general_model_path}")
        return None
    
    try:
        checkpoint = torch.load(general_model_path, weights_only=True)
        
        # 读取模型配置
        model_embed_dim = checkpoint.get('embedding_dim', embedding_dim)
        gamma = checkpoint.get('gamma', preset_best_param.get('gamma', 5.0))
        beta = checkpoint.get('beta', preset_best_param.get('beta', -10.0))
        centrality_scaling = checkpoint.get('centrality_scaling', 1.0)
        mixing_weight = checkpoint.get('mixing_weight', 0.5)
        noise_threshold = checkpoint.get('noise_threshold', 0.05)
        
        logging.info(f"加载模型参数: gamma={gamma:.4f}, beta={beta:.4f}, centrality_scaling={centrality_scaling:.4f}")
        logging.info(f"其他参数: mixing_weight={mixing_weight:.4f}, noise_threshold={noise_threshold:.4f}")
        
        # 创建新模型
        model = GKCIModel(
            G,
            embedding_dim=embedding_dim,
            hidden_dims=preset_best_param.get('hidden_dims', [64, 32]),
            num_layers=preset_best_param.get('num_layers', 1),
            num_sampled_neighbors=preset_best_param.get('num_sampled_neighbors', 20),
            dropout_rate=preset_best_param.get('dropout_rate', 0.1),
            use_batch_norm=preset_best_param.get('batch_norm', True)
        )
        
        # 如果读取的模型和创建的嵌入向量的维度不匹配，进行增强的特征适应
        if model_embed_dim != embedding_dim and transfer_learning_settings['feature_adaptation']:
            logging.warning(f"模型嵌入维度 ({model_embed_dim}) 与当前嵌入维度 ({embedding_dim}) 不匹配")
            logging.info("使用增强的特征适应层进行转换")
            
            # 创建更复杂的适配层，不仅是线性映射
            class EnhancedEmbeddingAdapter(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super(EnhancedEmbeddingAdapter, self).__init__()
                    self.linear1 = nn.Linear(input_dim, (input_dim + output_dim) // 2)
                    self.act = nn.LeakyReLU(0.2)
                    self.bn = nn.BatchNorm1d((input_dim + output_dim) // 2)
                    self.linear2 = nn.Linear((input_dim + output_dim) // 2, output_dim)
                    
                    # 初始化为近似身份映射
                    nn.init.xavier_uniform_(self.linear1.weight)
                    nn.init.xavier_uniform_(self.linear2.weight)
                    
                def forward(self, x):
                    x = self.linear1(x)
                    x = self.bn(x)
                    x = self.act(x)
                    x = self.linear2(x)
                    return x
            
            model.embed_adapter = EnhancedEmbeddingAdapter(embedding_dim, model_embed_dim)
            
            # 修改forward方法以使用适配层
            original_forward = model.forward
            
            def new_forward(embeddings, training=True):
                adapted_embeddings = model.embed_adapter(embeddings)
                return original_forward(adapted_embeddings, training)
            
            model.forward = new_forward
            
            # 此时我们可以加载原始模型的权重
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # 维度匹配，直接加载
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 设置参数值 - 使用加载的参数而不是模型默认值
        with torch.no_grad():
            model.gamma.copy_(torch.tensor(gamma))
            model.beta.copy_(torch.tensor(beta))
            
            # 设置新参数
            if hasattr(model, 'centrality_scaling'):
                model.centrality_scaling.copy_(torch.tensor(centrality_scaling))
            
            if hasattr(model, 'mixing_weight_raw'):
                # 反向计算raw值，使得sigmoid后等于目标值
                raw_value = torch.log(torch.tensor(mixing_weight) / (1 - torch.tensor(mixing_weight) + 1e-8))
                model.mixing_weight_raw.copy_(raw_value)
            
            if hasattr(model, 'noise_threshold'):
                # 反向计算raw值，使得sigmoid后等于目标值
                raw_value = torch.log(torch.tensor(noise_threshold) / (1 - torch.tensor(noise_threshold) + 1e-8))
                model.noise_threshold.copy_(raw_value)
        
        logging.info("通用模型加载成功，可用于迁移学习")
        return model
        
    except Exception as e:
        logging.error(f"加载通用模型失败: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return None


# -----------------------------------
# 主函数
# -----------------------------------
def main():
    # -----------------------------------
    # 1. 数据处理
    # -----------------------------------
    G, node_mapping = generate_sample_graph()  # 1.1 类依赖网络建模
    labels, key_nodes = get_key_class_labels(G, node_mapping)  # 获取关键类标签
    logging.info(f"关键类节点: {key_nodes}")
    
    # 1.2 计算节点嵌入  得到节点向量(embeddings)
    embeddings = compute_node_embeddings(G, dimensions=Node2Vec_dimensions)
    
    # 使用额外的中心性指标增强特征
    if use_multiple_centrality:
        centrality_features = compute_centrality_features(G)
        combined_features = combine_features(embeddings, centrality_features)
        logging.info(f"使用组合特征: 嵌入维度={embeddings.shape[1]}, 中心性特征维度={centrality_features.shape[1]}, 总维度={combined_features.shape[1]}")
    else:
        combined_features = combine_features(embeddings)
        logging.info(f"仅使用嵌入特征: 维度={combined_features.shape[1]}")

    # 使用迁移学习--加载通用模型general_model
    general_model = None
    if enable_transfer_learning:
        general_model = load_general_model(G, combined_features.shape[1])
        if general_model:
            logging.info("将使用迁移学习进行训练")
        else:
            logging.info("未找到通用模型，将创建新模型训练")

    # 最新模型数据路径use_previous_model
    prev_model_path = None
    if use_previous_model and not general_model:
        prev_model_path, last_train_id = load_latest_model()
        if prev_model_path:
            logging.info(f"将使用上次训练的模型继续训练: {prev_model_path}")
        else:
            logging.info("未找到之前的模型，将创建新模型训练")

    # -----------------------------------
    # 2. 模型构建 & 4. 优化与扩展
    # -----------------------------------
    # 参数调优否  参数best_param
    if searching_best_param:
        best_params = parameter_tuning(G, labels, combined_features)  # 参数调优
        logging.info("\n使用最佳参数进行训练和评估...")
    else:
        # 使用预设的参数
        best_params = preset_best_param
        logging.info("\n使用预设参数进行训练和评估...")

    
    k_list = [5, 10, 15]
    
    # K折交叉验证
    if num_splits > 1:
        logging.info(f"使用 {num_splits} 次划分进行交叉验证")
        
        # 将数据划分为训练集和测试集
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
        indices = np.arange(len(labels))
        
        # 存储每次划分的结果
        all_recalls = []
        all_precisions = []
        all_accuracy = []
        
        for split_idx, (train_indices, test_indices) in enumerate(kf.split(indices)):
            logging.info(f"\n训练和评估划分 {split_idx + 1}/{num_splits}")
            
            # 训练集和测试集标签初始化
            train_labels = np.zeros_like(labels)
            test_labels = np.zeros_like(labels)
            train_labels[train_indices] = labels[train_indices]
            test_labels[test_indices] = labels[test_indices]
            
            # 从训练集中划分验证集（20%）
            val_indices = train_indices[:len(train_indices)//5]
            actual_train_indices = train_indices[len(train_indices)//5:]
            val_labels = np.zeros_like(labels)
            val_labels[val_indices] = labels[val_indices]
            
            # 调整训练集，排除验证集
            actual_train_labels = np.zeros_like(labels)
            actual_train_labels[actual_train_indices] = labels[actual_train_indices]
            
            # 初始化并训练模型
            if general_model and split_idx == 0:
                # 仅在第一次划分时使用通用模型
                model = general_model
            else:
                model = GKCIModel(
                    G,
                    embedding_dim=combined_features.shape[1],
                    hidden_dims=best_params.get('hidden_dims', [64, 32]),
                    num_layers=best_params['num_layers'],
                    num_sampled_neighbors=best_params['num_sampled_neighbors'],
                    dropout_rate=best_params.get('dropout_rate', 0.3),
                    use_batch_norm=best_params.get('batch_norm', True)
                )
            
            # 使用验证集训练模型
            validation_data = (combined_features, val_labels)
            trained_model, history, optimizer, total_epochs = train_model(
                G,
                combined_features,
                actual_train_labels,
                num_epochs=num_epochs,
                learning_rate=best_params['learning_rate'],
                model=model,
                validation_data=validation_data
            )
            
            #  测试集评估
            trained_model.eval()
            embeddings_tensor = torch.tensor(combined_features, dtype=torch.float32)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            embeddings_tensor = embeddings_tensor.to(device)
            with torch.no_grad():
                final_scores = trained_model(embeddings_tensor, training=False)
            
            node_scores = final_scores.cpu().numpy()
            
            # 计算Recall@K（在前K个预测中正确识别关键类的比例）和Precision@K（前K个预测中实际是关键类的比例）
            test_recalls, test_precisions = compute_recall_precision_at_k(test_labels, node_scores, k_list)
            all_recalls.append(test_recalls)
            all_precisions.append(test_precisions)
            
            # 计算准确率（所有预测中正确的比例）
            preds = (node_scores > 0.5).astype(int)
            accuracy = np.sum(preds[test_indices] == test_labels[test_indices]) / len(test_indices)
            all_accuracy.append(accuracy)
            
            logging.info(f"划分 {split_idx + 1} 测试结果:")
            for k, recall, precision in zip(k_list, test_recalls, test_precisions):
                logging.info(f"K={k}: Recall={recall:.4f}, Precision={precision:.4f}")
            logging.info(f"分类准确率: {accuracy * 100:.2f}%")
            
            # 保存该划分的模型../historicalModels/Train-{train_id}/split_{split_idx}_model.pth目录
            save_model(trained_model, optimizer, total_epochs, train_id, split_idx)
            
            # 如果是第一次划分，保存为通用模型
            if split_idx == 0:
                metadata = {
                    'dataset': dataset,
                    'num_nodes': len(G.nodes()),
                    'num_edges': len(G.edges()),
                    'num_key_nodes': len(key_nodes)
                }
                save_general_model(trained_model, optimizer, total_epochs, metadata)
        
        # 计算平均结果
        avg_recalls = np.mean(all_recalls, axis=0)
        avg_precisions = np.mean(all_precisions, axis=0)
        avg_accuracy = np.mean(all_accuracy)
        
        logging.info(f"\n{num_splits}次交叉验证平均结果:")
        for k, recall, precision in zip(k_list, avg_recalls, avg_precisions):
            logging.info(f"K={k}: Avg Recall={recall:.4f}, Avg Precision={precision:.4f}")
        logging.info(f"平均分类准确率: {avg_accuracy:.4f}")
        
    else:
        # 单次训练模式
        logging.info("使用单次训练模式")
        
        # 8:2 划分训练集和测试集
        train_indices, test_indices = train_test_split(
            np.arange(len(labels)), 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        # 训练集和测试集标签初始化
        train_labels = np.zeros_like(labels)
        test_labels = np.zeros_like(labels)
        train_labels[train_indices] = labels[train_indices]
        test_labels[test_indices] = labels[test_indices]
        
        
        # 从训练集中提取1/5作为验证集
        val_indices = train_indices[:len(train_indices)//5]
        actual_train_indices = train_indices[len(train_indices)//5:]
        
        # 验证集标签初始化
        val_labels = np.zeros_like(labels)
        val_labels[val_indices] = labels[val_indices]
        actual_train_labels = np.zeros_like(labels)
        actual_train_labels[actual_train_indices] = labels[actual_train_indices]

        # 验证集训练模型
        model = general_model if general_model else None
        validation_data = (combined_features, val_labels)
        trained_model, history, optimizer, total_epochs = train_model(
            G,
            combined_features,
            actual_train_labels,
            num_epochs=num_epochs,
            learning_rate=best_params['learning_rate'],
            model=model,
            prev_model_path=prev_model_path,
            validation_data=validation_data
        )
        
        # 测试集评估模型
        trained_model.eval()
        embeddings_tensor = torch.tensor(combined_features, dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings_tensor = embeddings_tensor.to(device)
        with torch.no_grad():
            final_scores = trained_model(embeddings_tensor, training=False)
        
        node_scores = final_scores.cpu().numpy()
        
        # 计算测试集上的评价指标
        test_recalls, test_precisions = compute_recall_precision_at_k(test_labels, node_scores, k_list)
        
        logging.info(f"测试集结果:")
        for k, recall, precision in zip(k_list, test_recalls, test_precisions):
            logging.info(f"K={k}: Recall={recall:.4f}, Precision={precision:.4f}")
        
        # 计算准确率
        preds = (node_scores > 0.5).astype(int)
        accuracy = np.sum(preds[test_indices] == test_labels[test_indices]) / len(test_indices)
        logging.info(f"测试集分类准确率: {accuracy * 100:.2f}%")
        
        # 保存模型
        save_model(trained_model, optimizer, total_epochs, train_id)
        
        # 保存为通用模型
        metadata = {
            'dataset': dataset,
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges()),
            'num_key_nodes': len(key_nodes)
        }
        save_general_model(trained_model, optimizer, total_epochs, metadata)
    
    # 输出排名前K的重要类
    sorted_indices = np.argsort(node_scores)[::-1]  # 按分数从高到低排序
    print(f"\n识别出的重要类:")
    logging.info(f"\n识别出的重要类:")
    print("-" * 80)
    logging.info("-" * 80)
    print("{:<6} {:<8} {:<10} {:<6} {:<50}".format("排名", "节点ID", "分数", "是关键类", "类名"))
    logging.info("{:<6} {:<8} {:<10} {:<6} {:<50}".format("排名", "节点ID", "分数", "是关键类", "类名"))
    print("-" * 80)
    logging.info("-" * 80)
    for rank, idx in enumerate(sorted_indices[:15]):  # 显示前15个
        node_id = idx
        score = node_scores[idx]
        class_name = node_mapping[node_id] if node_mapping and node_id in node_mapping else f"Node-{node_id}"
        is_key = "✓" if labels[idx] == 1 else " "
        print("{:<6} {:<8} {:<10.4f} {:<6} {:<50}".format(rank+1, node_id, score, is_key, class_name))
        logging.info("{:<6} {:<8} {:<10.4f} {:<6} {:<50}".format(rank+1, node_id, score, is_key, class_name))
    print("-" * 80)
    logging.info("-" * 80)
    
    # 输出预测正确的关键类
    true_positives = []
    for idx in sorted_indices[:max(k_list)]:
        if labels[idx] == 1:
            true_positives.append(idx)
    
    if true_positives:
        print("\n成功识别的关键类:")
        logging.info("\n成功识别的关键类:")
        print("-" * 80)
        logging.info("-" * 80)
        for i, node_id in enumerate(true_positives):
            class_name = node_mapping[node_id] if node_mapping and node_id in node_mapping else f"Node-{node_id}"
            score = node_scores[node_id]
            rank = np.where(sorted_indices == node_id)[0][0] + 1
            log_line = f"{i+1}. {class_name} (排名: {rank}, 分数: {score:.4f}, 节点ID: {node_id})"
            print(log_line)
            logging.info(log_line)
        print("-" * 80)
        logging.info("-" * 80)
    
    # 计算在前K个中有多少正确识别的关键类
    for k in k_list:
        top_k_indices = sorted_indices[:k]
        correct_keys = sum(1 for idx in top_k_indices if labels[idx] == 1)
        total_keys = sum(labels)
        print(f"在前{k}个预测中，正确识别了{correct_keys}个关键类，共{total_keys}个关键类")
        logging.info(f"在前{k}个预测中，正确识别了{correct_keys}个关键类，共{total_keys}个关键类")
        
        # 计算覆盖率和准确率
        coverage = correct_keys / total_keys if total_keys > 0 else 0
        accuracy = correct_keys / k if k > 0 else 0
        print(f"覆盖率: {coverage:.2%}, 准确率: {accuracy* 100:.2f}%")
        logging.info(f"覆盖率: {coverage:.2%}, 准确率: {accuracy* 100:.2f}%")
    
    # 绘制loss和accuracy趋势图
    plot_training_history(history, title="Trend of Loss and Accuracy changes")
    
    # 在测试函数末尾添加保存分数的代码
    score_dir = os.path.join(result_dir, 'Scores')
    if not os.path.exists(score_dir):
        os.makedirs(score_dir)
    
    # 使用模型预测的分数而不是标签
    class_scores = node_scores
    
    # 保存分数到文件
    score_file_path = os.path.join(score_dir, f'{dataset}_scores.txt')
    with open(score_file_path, 'w') as f:
        for class_idx, score in enumerate(class_scores):
            f.write(f"{class_idx}--{score:.4f}\n")
        f.write("result_file_path: " + result_file_path + "\n")
    
    logging.info(f"分数已保存到 {score_file_path}")
    # 输出分数的统计信息
    min_score = np.min(class_scores)
    max_score = np.max(class_scores)
    mean_score = np.mean(class_scores)
    zero_scores = np.sum(class_scores < 0.051)
    logging.info(f"分数统计：最小值={min_score:.4f}, 最大值={max_score:.4f}, 平均值={mean_score:.4f}")
    logging.info(f"接近0的分数(小于0.051)数量: {zero_scores} / {len(class_scores)}")


# 运行主函数并将输出重定向到结果文件
if __name__ == '__main__':
    main()
    logging.info(f"log已保存到 {result_file_path}")
