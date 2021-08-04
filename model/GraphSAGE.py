import sys, os
import torch
import random
import numpy as np 
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
import torchsnooper

#@torchsnooper.snoop()
class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, input_feature_dim, output_feature_dim): 
        super(SageLayer, self).__init__()

        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim

        self.weight = nn.Parameter(torch.FloatTensor(self.output_feature_dim, self.input_feature_dim))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.normal_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.
        nodes     -- list of nodes
        """
        combined = torch.cat([self_feats, aggregate_feats], dim=1)
        
        combined = F.relu(self.weight.mm(combined.t())).t() 
        return combined


#@torchsnooper.snoop()
class GraphSage(nn.Module):
    """docstring for GraphSage"""
    def __init__(self, 
                 input_feature_dim, 
                 output_feature_dim,
                 node_features, 
                 edge_features, 
                 adj_lists,
                 edgeid_to_idx, 
                 device, 
                 num_layers=2):
        """
        init
        """
        super(GraphSage, self).__init__()

        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        self.num_layers = num_layers
        self.device = device

        self.node_features = node_features
        self.edge_features = edge_features
        self.adj_lists = adj_lists
        self.edgeid_to_idx = edgeid_to_idx

        self.sage_layer1 = SageLayer(self.input_feature_dim, self.output_feature_dim)
        self.sage_layer2 = SageLayer(self.input_feature_dim, self.output_feature_dim)

    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch    -- batch of nodes to learn the embeddings
        """
        upper_layer_nodes = np.array(nodes_batch)
        nodes_batch_layers = [(upper_layer_nodes,)]

        for i in range(self.num_layers):
            lower_layer_nodes, samp_edges = self.get_lower_layer(upper_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, samp_edges))
            upper_layer_nodes = lower_layer_nodes

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.node_features
        for layer_idx in range(1, self.num_layers+1):
            upper_layer_nodes_idx, lower_layer = nodes_batch_layers[layer_idx][0],nodes_batch_layers[layer_idx-1]

            aggregate_feats = self.aggregate(upper_layer_nodes_idx, lower_layer)

            if layer_idx > 1:
                upper_layer_nodes_idx = self._nodes_map(upper_layer_nodes_idx, lower_layer)
            
            sage_layer = getattr(self, 'sage_layer'+str(layer_idx))
            cur_hidden_embs = sage_layer(self_feats = pre_hidden_embs[upper_layer_nodes_idx], aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def get_reverse_adj_list(self):
        adj_lists_r = defaultdict(set)
        for k,vs in self.adj_lists.items():
            for v in vs:
                adj_lists_r[v].add(k)
        return adj_lists_r
        
    def _nodes_map(self, nodes, lower_layer):
        unique_from_index, samp_edges = lower_layer
        i = list(range(len(unique_from_index)))
        unique_nodes = dict(list(zip(unique_from_index, i)))
        
        index = np.array([unique_nodes[x] for x in nodes])
        return index
        
    def get_lower_layer(self, nodes_batch, sample_num=10):
        """
        Args:
            nodes_batch: extended node batch
            sample_num: sample threshold
            
        Returns:
            unique_from_index: (n_nodes,) int tentor 
            samp_edges: [[from_node_idx,to_node_idx],...] (n_edges,2) int tentor
        """
        adj_lists_r = self.get_reverse_adj_list()
        to_neighs_list = [adj_lists_r[int(node)] for node in nodes_batch]
        samp_neighs_list = [set(random.sample(to_neighs, sample_num)) if len(to_neighs) >= sample_num else to_neighs for to_neighs in to_neighs_list]
        # 一些无邻接的结点 可能本身被采样掉了
        samp_neighs_list = [samp_neighs | set([nodes_batch[i]]) for i, samp_neighs in enumerate(samp_neighs_list)]
        samp_edges = np.array([[samp_neigh, nodes_batch[i]] for i, samp_neighs in enumerate(samp_neighs_list) for samp_neigh in samp_neighs])
        
        from_index, to_index = samp_edges[:,0], samp_edges[:,1]
        unique_from_index = np.unique(from_index)
        return unique_from_index, samp_edges
    
    def aggregate(self, upper_layer_nodes, lower_layer):
        unique_from_index, samp_edges = lower_layer
        #edge_index = np.vstack([from_index, to_index]).T
        from_index, to_index = samp_edges[:,0], samp_edges[:,1]
        
        from_states = self.node_features[from_index]
        #to_states = self.node_features[to_index]
        edge_states = self.edge_features[[self.edgeid_to_idx[tuple(i)] for i in samp_edges]]
        
        # print(from_states.shape,to_states.shape,edge_states.shape)
        messages = torch.cat([from_states, edge_states], dim=1)
        
        agg_messages = self._unsorted_segment_sum(messages, to_index, upper_layer_nodes)
        return agg_messages

    def _unsorted_segment_sum(self, data, segment_ids, nodes):
        tensor = torch.zeros(len(nodes), data.shape[1])
        for index in range(len(nodes)):
            tensor[index, :] = torch.sum(data[segment_ids == nodes[index], :], dim=0)
        return tensor

# @torchsnooper.snoop()
class UnsupervisedLoss(object):
    """docstring for UnsupervisedLoss"""
    def __init__(self, adj_lists, train_nodes, device):
        super(UnsupervisedLoss, self).__init__()
        self.Q = 10
        self.N_WALKS = 6
        self.WALK_LEN = 1
        self.N_WALK_LEN = 5
        self.MARGIN = 3
        self.adj_lists = adj_lists
        self.train_nodes = train_nodes
        self.device = device

        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []

    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q*torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            #print(neg_score)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            #print(pos_score)

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1,-1))
                
        loss = torch.mean(torch.cat(nodes_score, 0))
        
        return loss
        
    def get_loss_margin(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i]==self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n:i for i,n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

            nodes_score.append(torch.max(torch.tensor(0.0).to(self.device), neg_score-pos_score+self.MARGIN).view(1,-1))
            # nodes_score.append((-pos_score - neg_score).view(1,-1))

        loss = torch.mean(torch.cat(nodes_score, 0),0)

        # loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))

        return loss
 
    def extend_nodes(self, nodes, num_neg=6):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        self.node_negtive_pairs = {}

        self.target_nodes = nodes
        self.get_positive_nodes(nodes)
        # print(self.positive_pairs)
        self.get_negtive_nodes(nodes, num_neg)
        # print(self.negtive_pairs)
        self.unique_nodes_batch = list(set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
        assert set(self.target_nodes) < set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_positive_nodes(self, nodes):
        return self._run_random_walks(nodes)

    def get_negtive_nodes(self, nodes, num_neg):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train_nodes) - neighbors
            neg_samples = random.sample(far_nodes, num_neg) if num_neg < len(far_nodes) else far_nodes
            self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    def _run_random_walks(self, nodes):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                continue
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    neighs = self.adj_lists[int(curr_node)]
                    next_node = random.choice(list(neighs))
                    # self co-occurrences are useless
                    if next_node != node and next_node in self.train_nodes:
                        self.positive_pairs.append((node,next_node))
                        cur_pairs.append((node,next_node))
                    curr_node = next_node

            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs

# @torchsnooper.snoop()
class Classification(nn.Module):

	def __init__(self, emb_size, num_classes):
		super(Classification, self).__init__()

		#self.weight = nn.Parameter(torch.FloatTensor(emb_size, num_classes))
		self.layer = nn.Sequential(
								nn.Linear(emb_size, num_classes)	  
								#nn.ReLU()
							)
		self.init_params()

	def init_params(self):
		for param in self.parameters():
			if len(param.size()) == 2:
				nn.init.xavier_uniform_(param)

	def forward(self, embeds):
		logists = torch.log_softmax(self.layer(embeds), 1)
		return logists