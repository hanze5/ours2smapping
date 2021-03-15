import numpy as np
import scipy.sparse as sp

INF = 0x3f3f3f


class Graph():
    '''
    读出查询图并用邻接表储存
    '''
    def __init__(self, file_path):
        self.graph = None
        self.adjacency_list = None
        self.read_in_graph(file_path)
        self.init_adjacency_list()

    def init_adjacency_list(self):     ##初始化邻接表  默认每个节点的度不超过4个？
            node_num = len(self.graph)
            self.adjacency_list = []
            for i in range(node_num):
                self.adjacency_list.append([])
            for i in range(node_num):
                for j in range(node_num):
                    if i == j:
                        continue
                    if j + 1 in self.graph[i] or i + 1 in self.graph[j]: # or for undirected graph
                        self.adjacency_list[i].append(j)

    def read_in_graph(self, file_path):
        graph = []
        with open(file_path, 'r') as f:
            line = f.readline()
            line = line.split(" output ")[0]

            node_vectors = line.split(", ")

            for idx in range(len(node_vectors)):
                child_nodes = node_vectors[idx].split(",")
                graph.append([int(child) for child in child_nodes])   ##child_nodes的int型式
        self.graph = graph


class TargetGraph(Graph):
    def __init__(self, file_path):
        '''
        实例化self.dt为结点之间的最短路径
        '''
        super(TargetGraph, self).__init__(file_path)
        self.dt = None
        self.init_distance_table()
        self.compute_min_distance()


    def init_distance_table(self):
        graph = self.graph    
        node_num = len(graph)
        dt = np.zeros([node_num, node_num], dtype=int)       ##设置距离矩阵，初始化距离都为0
        for i in range(node_num):
            for j in range(node_num):
                if i == j:                                   ##节点距离自己的距离为0
                    dt[i][j] = 0
                elif j + 1 in graph[i] or i + 1 in graph[j]: # or for undirected graph  有边则设为1，无边则设为无穷
                    dt[i][j] = 1
                else:
                    dt[i][j] = INF
        self.dt = dt

    # use floyd algorithm to calculate shortest path         
    def compute_min_distance(self):
        '''
        迭代更新dt，最终储存结点之间的最短路径
        '''
        size = len(self.graph)
        for k in range(size):
            for i in range(size):
                for j in range(size):
                    if self.dt[i][j] > self.dt[i][k] + self.dt[k][j]:
                        self.dt[i][j] = self.dt[i][k] + self.dt[k][j]

    def get_grf_size(self):
        '''
        返回图的规模
        '''
        return len(self.graph)

    # all start from 1
    def get_distance_between(self, source, target):
        '''
        返回查询图的特定结点到目标图结点之间的距离？？    返回图中任意两结点之间的最短距离
        '''
        assert (source > 0 and target > 0)
        return float(self.dt[source - 1][target - 1])



    # inputs: containing source graph msg       
    # preceeding_action_list: a vector of partial mapping named 'core set' in  VF2, 
    '''
    已匹配的映射集合？ 核心集M（s）
    '''
    #       the first element give the mapping in target graph for node 1, second for node 2 etc.
    #       target graph index start from 0
    # return: a list of target nodes that's feasible, node idx starts from 0
 
    def get_infeasible_mapping_node(self, source_graph, preceeding_action_list, is_init):
        '''
        获取不可匹配的节点列表？？
        '''
        def check_r_pred_succ(target_node):
            '''
            检查是否匹配成功
            '''
            '''
            重要前提，当前结点之前的是一定在核心集里的，否则不可能执行到当前结点
            '''
            #1.2   (也是最初版本的仅有约束)
            for node in source_graph.adjacency_list[len(preceeding_action_list)]: 
                #node结点是候选结点的邻居结点
                if node < len(preceeding_action_list): # already in core set
                    if preceeding_action_list[node] not in self.adjacency_list[target_node]:
                        return False
            #1.3
            for node in self.adjacency_list[target_node]:
                #node结点是target结点的邻居结点
                if node in preceeding_action_list: # already in core set
                    if  preceeding_action_list.index(node) not in source_graph.adjacency_list[len(preceeding_action_list)]:
                        return False
            #2
            small_nb_of_TandM = 0
            big_nb_of_TandM = 0

            for node in source_graph.adjacency_list[len(preceeding_action_list)]: 
                #node结点是候选节点的邻居结点
                for snode in source_graph.adjacency_list[node]:
                    #snode是候选结点邻居结点的邻居结点
                    if snode< len(preceeding_action_list): # already in core set
                        small_nb_of_TandM += 1
                        break

            for node in self.adjacency_list[target_node]:
                #node是尝试匹配的target结点的邻居结点
                for bnode in self.adjacency_list[node]:
                    #bnode是尝试匹配的target结点的邻居节点
                    if bnode in preceeding_action_list:
                        big_nb_of_TandM += 1
                        break
            if (small_nb_of_TandM > big_nb_of_TandM):
                return False

            #3     
            small_nb_of_TandM = 0
            big_nb_of_TandM = 0
            for node in source_graph.adjacency_list[len(preceeding_action_list)]: 
                #node结点是候选节点的邻居结点
                for snode in source_graph.adjacency_list[node]:
                    #snode是候选结点邻居结点的邻居结点
                    if snode> len(preceeding_action_list): # already in core set
                        small_nb_of_TandM += 1
                        break

            for node in self.adjacency_list[target_node]:
                #node是尝试匹配的target结点的邻居结点
                for bnode in self.adjacency_list[node]:
                    #bnode是尝试匹配的target结点的邻居节点
                    if bnode not in preceeding_action_list:
                        big_nb_of_TandM += 1
                        break
            if (small_nb_of_TandM > big_nb_of_TandM):
                return False

            return True

        # 结点分为三种模式，初始化为3，正在操作以及操作过的结点为1，与1结点相邻的结点为2
        if is_init:
            self.set_vector_source = [3] * len(source_graph.graph)  #查询图规模的全3
            # print("First: ", self.set_vector_source)
            self.set_vector_target = [3] * len(self.graph)          #目标图规模的全3
            self.current_source_node = -1
            # self.feasible_set = []
            infeasible = []
        else:
            # core set = preceeding list
            self.set_vector_source[self.current_source_node] = 1    #将当前操作的结点模式设为1
            # print("Second: ", self.set_vector_source)
            self.set_vector_target[preceeding_action_list[self.current_source_node]] = 1   #已被匹配到的目图中的结点模式也设为1
            # feasible_set.remove(preceeding_action_list[self.current_source_node])

            #feasible set
            for neighbour in self.adjacency_list[preceeding_action_list[self.current_source_node]]:  #被匹配到的目标图中邻居未被操作过的全部设2
                if self.set_vector_target[neighbour] != 1:
                    self.set_vector_target[neighbour] = 2
                    # feasible_set.append(neighbour)

            for neighbour in source_graph.adjacency_list[self.current_source_node]:          #查询图中同上操作
                if self.set_vector_source[neighbour] != 1:
                    self.set_vector_source[neighbour] = 2

            infeasible = []
            ##挨个判断 
            for node in range(len(self.graph)):                    
                if node in preceeding_action_list \
                        or self.set_vector_target[node] != 2 \
                        or check_r_pred_succ(node) == False:
                    infeasible.append(node)

        self.current_source_node += 1
        return infeasible

class SourceGraph(Graph):
    def __init__(self, file_path):
        super(SourceGraph, self).__init__(file_path)

class ReadingGraph(Graph):
    '''
    邻接矩阵
    '''
    def __init__(self, file_path):
        super(ReadingGraph, self).__init__(file_path)
        self.dt = None
        self.adjacency_matrix()

    def adjacency_matrix(self):
        graph = self.graph
        node_num = len(graph)
        dt = np.zeros([node_num, node_num], dtype=int)
        for i in range(node_num):
            for j in range(node_num):
                if i == j:
                    dt[i][j] = 0
                elif j+1 in graph[i] or i+1 in graph[j]:
                     dt[i][j] = 1
        self.dt = dt