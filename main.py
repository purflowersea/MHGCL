import os
import argparse
import gc
import torch
from tqdm import tqdm
from rdkit import Chem
import numpy as np
import json
import copy
from utils import *
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from model.mhgcl import MHGCL
from torch_geometric.utils import degree
from torch.utils.data.distributed import DistributedSampler
from data_process import smile_to_graph, read_smiles, read_interactions, generate_node_subgraphs, read_network, read_targets, load_protein_embeddings
from sklearn.model_selection import StratifiedKFold, KFold
from train_eval import train, test, eval
from weight_analysis import visualize_weights, analyze_optimal_weights, record_weights_in_training, analyze_all_folds
from optimal_weights import get_optimal_weights

import random
from data_process import load_id_mapping

def init_args(user_args=None):

    parser = argparse.ArgumentParser(description='MHGCL')

    parser.add_argument('--model_name', type=str, default='mhgcl')

    parser.add_argument('--dataset', type=str, default="drugbank")

    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--layer', type=int, default=2)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--model_episodes', type=int, default=100)
    
    parser.add_argument('--extractor', type=str, default="adaptive",
                       choices=["khop-subtree", "randomWalk", "probability", "adaptive"],
                       help="Subgraph extraction method (adaptive:adaptive method with auto-tuning)")
    parser.add_argument('--graph_fixed_num', type=int, default=1)
    parser.add_argument('--khop', type=int, default=2)
    parser.add_argument('--fixed_num', type=int, default=32)

    # Graphormer
    parser.add_argument("--d_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_smiles_degree", type=int, default=300)
    parser.add_argument("--max_graph_degree", type=int, default=600)
    parser.add_argument("--dropout", type=float, default=0.2)
    
    # 模态权重参数
    parser.add_argument('--knowledge_weight', type=float, default=0.25,
                        help='Weight for knowledge graph modality')
    parser.add_argument('--molecular_weight', type=float, default=0.25,
                        help='Weight for molecular graph modality')
    parser.add_argument('--smiles_weight', type=float, default=0.25,
                        help='Weight for SMILES sequence modality')
    parser.add_argument('--target_weight', type=float, default=0.25,
                        help='Weight for target information modality')
    parser.add_argument('--fixed_weights', action='store_true',
                        help='Use fixed weights instead of dynamic weights for modality fusion')
    

    # 对比学习参数
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                        help='Weight for contrastive learning')
    parser.add_argument('--mol_ratio', type=float, default=0.5,
                        help='Ratio for molecular contrastive learning')
    parser.add_argument('--kg_ratio', type=float, default=0.5,
                        help='Ratio for knowledge graph contrastive learning')
    parser.add_argument('--mol_temperature', type=float, default=0.1,
                        help='Temperature parameter for molecular contrastive learning')
    parser.add_argument('--kg_temperature', type=float, default=0.2,
                        help='Temperature parameter for knowledge graph contrastive learning')
    parser.add_argument('--mol_aug_ratio', type=float, default=0.1,
                        help='Augmentation ratio for molecular graphs')
    parser.add_argument('--kg_aug_ratio', type=float, default=0.3,
                        help='Augmentation ratio for knowledge graphs')
    parser.add_argument('--fixed_contrastive', action='store_false',
                        help='Use fixed contrastive learning parameters')
    
    # coeff
    parser.add_argument('--sub_coeff', type=float, default=0.1)
    parser.add_argument('--mi_coeff', type=float, default=0.1)

    parser.add_argument('--s_type', type=str, default='random')

    parser.add_argument('--fusion_temperature', type=float, default=1.0,
                    help='Temperature parameter for fusion module')

    args = parser.parse_args()

    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def k_fold(data, kf, folds, y):

    test_indices = []
    train_indices = []

    if len(y):
        for _, idx in kf.split(torch.zeros(len(data)), y):
            test_indices.append(idx)
    else:
        for _, idx in kf.split(data):
            test_indices.append(idx)

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(data), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices

def split_fold(folds, dataset, labels, scenario_type='random'):

    test_indices, train_indices, val_indices = [], [], []

    if scenario_type == 'random':##这是根据interactions在划分的数据集，也就是根据interactions的label进行的数据集划分
        skf = StratifiedKFold(folds, shuffle=True, random_state=2023)
        train_indices, test_indices, val_indices = k_fold(dataset, skf, folds, labels)

    return train_indices, test_indices, val_indices

def load_data(args):

    dataset = args.dataset

    data_path = "dataset/" + dataset + "/"

    numid_to_drugid = load_id_mapping(os.path.join(data_path, "drug_smiles_data.csv"))
    
    # **1. 读取 SMILES 数据**
    ligands = read_smiles(os.path.join(data_path, "drug_smiles.txt"))
    
    # **2. 加载ESM-2嵌入并使用 读取 靶点数据**
    esm2_embeddings = load_protein_embeddings(os.path.join(data_path, "drugbank_target_embeddings.pt"))
    target_dict = read_targets(os.path.join(data_path, "drug_target_data.csv"), esm2_embeddings=esm2_embeddings)

    # smiles to graphs
    # **3. 生成分子图**
    print("load drug smiles graphs!!")
    # 生成分子图，同时保存 SMILES 序列
    smile_graph, num_rel_mol_update, max_smiles_degree = smile_to_graph(data_path, ligands)

    print("load networks !!")
    num_node, network_edge_index, network_rel_index, num_rel = read_network(data_path + "networks.txt")

    print("load DDI samples!!")
    interactions_label, all_contained_drgus = read_interactions(os.path.join(data_path, "ddi.txt"), smile_graph)
    interactions = interactions_label[:, :2]
    labels = interactions_label[:, 3]


    print("generate subgraphs!!")
    drug_subgraphs, max_subgraph_degree, num_rel_update = generate_node_subgraphs(dataset, all_contained_drgus,
                                                                                  network_edge_index, network_rel_index,
                                                                                  num_rel, args)

    data_sta = {
        'num_nodes': num_node + 1,
        'num_rel_mol': num_rel_mol_update + 1,
        'num_rel_graph': num_rel_update + 1,
        'num_interactions': len(interactions),
        'num_drugs_DDI': len(all_contained_drgus),
        'max_degree_graph': max_smiles_degree + 1,
        'max_degree_node': int(max_subgraph_degree)+1
    }

    print(data_sta)

    return interactions, labels, smile_graph, drug_subgraphs, target_dict, data_sta, numid_to_drugid

def save(save_dir, args, train_log, test_log):
    args.device = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(save_dir + "/args.json", 'w') as f:
        json.dump(args.__dict__, f)
    with open(save_dir + '/test_results.json', 'w') as f:
        json.dump(test_log, f)
    with open(save_dir + '/train_log.json', 'w') as f:
        json.dump(train_log, f)

def save_results(save_dir, args, results_list):
    acc = []
    auc = []
    aupr = []
    f1 = []

    for r in results_list:
        acc.append(r['acc'])
        auc.append(r['auc'])
        aupr.append(r['aupr'])
        f1.append(r['f1'])

    acc = np.array(acc)
    auc = np.array(auc)
    aupr = np.array(aupr)
    f1 = np.array(f1)

    results = {
        'acc':[np.mean(acc),np.std(acc)],
        'auc':[np.mean(auc),np.std(auc)],
        'aupr': [np.mean(aupr), np.std(aupr)],
        'f1': [np.mean(f1), np.std(f1)],
    }

    args = vars(args)
    args.update(results)

    with open(save_dir + args['extractor'] + '_all_results.json', 'a+') as f:
        json.dump(args, f)


def init_model(args, dataset_statistics):
    if args.model_name == 'mhgcl':
        model = MHGCL(args=args,
                      max_layer=args.layer,
                      num_features_drug = 67,
                      num_nodes=dataset_statistics['num_nodes'],
                      num_relations_mol=dataset_statistics['num_rel_mol'],
                      num_relations_graph=dataset_statistics['num_rel_graph'],
                      output_dim=args.d_dim,
                      max_degree_graph=dataset_statistics['max_degree_graph'],
                      max_degree_node = dataset_statistics['max_degree_node'],
                      sub_coeff=args.sub_coeff,
                      mi_coeff=args.mi_coeff,
                      dropout=args.dropout,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      num_heads=args.num_heads,
                      fusion_temperature=args.fusion_temperature,
                      contrastive_weight=args.contrastive_weight,
                      mol_ratio=args.mol_ratio,
                      kg_ratio=args.kg_ratio
                      )


    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    return model, optimizer

def main(args = None, k_fold = 5):

    if args is None:
        args = init_args()

    results_of_each_fold = []

    ##加载interactions的data
    data, labels, smile_graph, node_graph, target_dict, dataset_statistics, numid_to_drugid = load_data(args)

    # 自动加载最优权重
    if args.fixed_weights:
        optimal_weights = get_optimal_weights(args.dataset)
        args.knowledge_weight = optimal_weights["knowledge_weight"]
        args.molecular_weight = optimal_weights["molecular_weight"]
        args.smiles_weight = optimal_weights["smiles_weight"]
        args.target_weight = optimal_weights["target_weight"]
        print(f"Using optimal fixed weights for dataset '{args.dataset}':")
        print(f"  - Knowledge weight: {args.knowledge_weight:.4f}")
        print(f"  - Molecular weight: {args.molecular_weight:.4f}")
        print(f"  - SMILES weight: {args.smiles_weight:.4f}")
        print(f"  - Target weight: {args.target_weight:.4f}")

    # 自动加载对比学习参数
    if args.fixed_contrastive:
        optimal_weights = get_optimal_weights(args.dataset)
        args.contrastive_weight = optimal_weights["contrastive_weight"]
        args.mol_ratio = optimal_weights["mol_ratio"]
        args.kg_ratio = optimal_weights["kg_ratio"]
        args.mol_temperature = optimal_weights["mol_temperature"]
        args.kg_temperature = optimal_weights["kg_temperature"]
        args.mol_aug_ratio = optimal_weights["mol_aug_ratio"]
        args.kg_aug_ratio = optimal_weights["kg_aug_ratio"]
        print(f"Using optimal contrastive learning parameters for dataset '{args.dataset}':")
        print(f"  - Contrastive weight: {args.contrastive_weight:.4f}")
        print(f"  - Molecular ratio: {args.mol_ratio:.4f}")
        print(f"  - KG ratio: {args.kg_ratio:.4f}")
        print(f"  - Molecular temperature: {args.mol_temperature:.4f}")
        print(f"  - KG temperature: {args.kg_temperature:.4f}")
        print(f"  - Molecular augmentation ratio: {args.mol_aug_ratio:.4f}")
        print(f"  - KG augmentation ratio: {args.kg_aug_ratio:.4f}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_seed(42)
        
    ##split datasets
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*split_fold(k_fold, data, labels, args.s_type))):
        print(f"============================{fold+1}/{k_fold}==================================")
        print("loading data!!")
        ##load_data
        train_data = DTADataset(x=data[train_idx], y=labels[train_idx], sub_graph=node_graph, smile_graph=smile_graph, target_dict=target_dict, numid_to_drugid=numid_to_drugid)
        test_data = DTADataset(x=data[test_idx], y=labels[test_idx], sub_graph=node_graph, smile_graph=smile_graph, target_dict=target_dict, numid_to_drugid=numid_to_drugid)
        eval_data = DTADataset(x=data[val_idx], y=labels[val_idx], sub_graph=node_graph, smile_graph=smile_graph, target_dict=target_dict, numid_to_drugid=numid_to_drugid)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)  ##用DataLoader加载的数据，index是会自动增加的！！
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)  ##用DataLoader加载的数据，index是会自动增加的！！
        eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)  ##用DataLoader加载的数据，index是会自动增加的！！
        
        # 初始化权重记录目录
        weight_records_dir = os.path.join('./weight_records/', args.model_name, args.dataset, args.extractor, f"fold_{fold}")
        os.makedirs(weight_records_dir, exist_ok=True)

        if args.model_name:
            model, optimizer = init_model(args, dataset_statistics)
            model.to(device)
            model.reset_parameters()

            ##train_model
            #trange = tqdm(range(1, args.model_episodes + 1))
            best_auc = 0.0
            early_stop_num = 0

            train_log = {'train_acc':[], 'train_auc':[], 'train_aupr':[], 'train_loss':[],
                         'eval_acc':[], 'eval_auc':[], 'eval_aupr':[], 'eval_loss':[],
                         'modal_weights':[]}  # 添加权重记录

            for i_episode in range(args.model_episodes):
                loop = tqdm(train_loader, ncols=80)
                loop.set_description(f'Epoch[{i_episode}/{args.model_episodes}]')
                # 修改接收返回值，增加modal_weights
                train_acc, train_f1, train_auc, train_aupr, train_loss, modal_weights = train(loop, model, optimizer)
                eval_acc, eval_f1, eval_auc, eval_aupr, eval_loss = eval(eval_loader, model)
                print(f"train_auc:{train_auc} train_aupr:{train_aupr} eval_auc:{eval_auc} eval_aupr:{eval_aupr}")
                
                # 使用 weight_analysis 中的函数记录权重
                train_log = record_weights_in_training(
                    train_log, 
                    modal_weights, 
                    i_episode, 
                    (eval_acc, eval_f1, eval_auc, eval_aupr)
                )

                train_log['train_acc'].append(train_acc)
                train_log['train_auc'].append(train_auc)
                train_log['train_aupr'].append(train_aupr)
                train_log['train_loss'].append(train_loss)

                train_log['eval_acc'].append(eval_acc)
                train_log['eval_auc'].append(eval_auc)
                train_log['eval_aupr'].append(eval_aupr)
                train_log['eval_loss'].append(eval_loss)
                
                # 定期生成权重可视化图表
                if (i_episode + 1) % 10 == 0 or i_episode == 0 or i_episode == args.model_episodes - 1:
                    if 'modal_weights' in train_log and train_log['modal_weights']:
                        visualize_weights(
                            train_log['modal_weights'], 
                            os.path.join(weight_records_dir, f'weights_until_epoch_{i_episode+1}.png')
                        )

                if eval_auc > best_auc:
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_auc = eval_auc
                    # 保存达到最佳AUC时的权重
                    if modal_weights:
                        best_weights = modal_weights
                    early_stop_num = 0
                else:
                    early_stop_num += 1
                    if early_stop_num > 20:
                        print("early stop!")
                        break
            
            # 训练结束后分析最优权重
            if 'modal_weights' in train_log and train_log['modal_weights']:
                optimal_weights_report = analyze_optimal_weights(
                    train_log['modal_weights'], 
                    weight_records_dir, 
                    eval_metric='eval_auc'
                )

            model.load_state_dict(best_model_state)
            model.to(device)
            test_log = test(test_loader, model) ##test_log是一个字典，里面存储着metrics
            
            # 更新args中的权重参数为最佳模型的权重
            if 'best_weights' in locals() and best_weights:
                args.knowledge_weight = best_weights['knowledge']
                args.molecular_weight = best_weights['molecular']  
                args.smiles_weight = best_weights['smiles']
                args.target_weight = best_weights['target']
                print(f"Best model weights - Knowledge: {args.knowledge_weight:.4f}, Molecular: {args.molecular_weight:.4f}, SMILES: {args.smiles_weight:.4f}, Target: {args.target_weight:.4f}")

            save_dir = os.path.join('./best_save/', args.model_name, args.dataset, args.extractor,
                                    "fold_{}".format(fold), "{:.5f} SMILES ESM-2 TARGET RHZT DBGDQZ kg0.2 mol0.15 2" .format(test_log['auc']))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            model.save(save_dir)  ##保存当前最好的模型
            save(save_dir, args, train_log, test_log)
            print(f"save to {save_dir}")
            results_of_each_fold.append(test_log)
            # 在保存完所有结果后，复制最优权重分析到最终保存目录
            if 'optimal_weights_report' in locals() and optimal_weights_report:
                with open(os.path.join(save_dir, 'optimal_weights_analysis.json'), 'w') as f:
                    json.dump(optimal_weights_report, f, indent=2)
    
    # 分析所有fold的权重数据
    analyze_all_folds(os.path.join('./weight_records/', args.model_name, args.dataset, args.extractor))
    save_results(os.path.join('./best_save/', args.model_name, args.dataset), args, results_of_each_fold)

    return;

if __name__ == "__main__":
    main()
