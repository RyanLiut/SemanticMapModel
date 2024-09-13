import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import matplotlib.font_manager as fm
fm.fontManager.addfont('times_with_simsun.ttf')
from sklearn.manifold import MDS

class SemanticMap:
    def __init__(self, tfM, featNames, adjM=None, GT_adj=None, zeroOcc=0):
        self.tfM = tfM # (N, D)
        self.featNames = {ix:i for ix, i in enumerate(featNames)}
        self.mergeFeat()
        self.GT_adj = GT_adj
        self.zeroOcc = zeroOcc
        self.adjM = adjM

    def mergeFeat(self):
        # merge the feature columns if they are totally the same
        unique_featNames = []
        unique_columns = []
        for i, col in enumerate(self.tfM.T.tolist()):
            if col not in unique_columns or sum(col) == 0:
                unique_columns.append(col)
                unique_featNames.append(self.featNames[i])
            else:
                ix = unique_columns.index(col)
                unique_featNames[ix] += "/" + self.featNames[i]
        self.featNames = {ix:i for ix, i in enumerate(unique_featNames)}
        self.tfM = np.stack(unique_columns,axis=1)
        print(self.featNames) # for debug
        print(self.tfM)

    def constG(self):
        '''
        To construct a undirected acyclic (fully connected) graph according to a term-feature matrix
        '''
        mat = self.tfM.T @ self.tfM + (1 - self.tfM).T @ (1 - self.tfM) * self.zeroOcc # 0 has the fastest convergence.
        print(mat.shape)
        if self.adjM is None:
            self.adjM = np.triu(mat)

        np.fill_diagonal(self.adjM, val=0) # (D, D)

        print(self.adjM)
        self.G = nx.from_numpy_array(self.adjM)
        # self.visualizeSM(self.G)
        for u, v in self.G.edges():
            weight = self.G.get_edge_data(u, v)['weight']
            # print(f"Edge ({u}, {v}) has weight {weight}.")
        components = list(nx.connected_components(self.G))
        print("There are %d connected components at the previous Graph"%(len(components)))
        if len(components) != 1: # Due to the zero-occurrence, the components are always connected.
            print(components)

    def accWithGT(self, T):
        return np.sum(nx.to_numpy_array(T) == self.GT_adj) / (self.GT_adj.shape[0] ** 2)
    
    def check_subGraph_connectivity(self, G, selected_ins):
        connected_flag = []
        self.subG_list = []
        for i in self.tfM[selected_ins, :]: # O(N)
            subG = G.subgraph({ix for ix,j in enumerate(i) if j==1})
            connected = nx.is_connected(subG)
            connected_flag.append(connected)
            self.subG_list.append(subG)

        acc = sum(connected_flag) / len(connected_flag)
        Deg = [d for _, d in G.degree()]
        Deg_mean = np.mean(Deg)
        Deg_std = np.std(Deg)

        return connected_flag, acc, Deg_mean, Deg_std

    def highlight_subgraphs(self, G, subgraphs, linestyles=None, colors=None):
        """ TODO: NOT COMPETE!
        Highlight connected subgraphs in a NetworkX graph by drawing lines around them.

        Args:
            G: The NetworkX graph.
            subgraphs: A list of subgraphs, where each subgraph is a list of nodes.
            linestyles: A list of linestyles for the bounding lines (optional).
            colors: A list of colors for the bounding lines (optional).
        """

        if linestyles is None:
            linestyles = ["solid"] * len(subgraphs)
        if colors is None:
            colors = ["red", "blue", "green"]  # Cycle through colors for visibility

        lay = nx.planar_layout(G)
        nx.set_node_attributes(G, lay, 'pos')  # Assign positions to the 'pos' attribute
        # Calculate bounding boxes for each subgraph
        bounding_boxes = []
        for subgraph in subgraphs:
            pos = nx.get_node_attributes(G, 'pos')
            xs = [pos[node][0] for node in subgraph]
            ys = [pos[node][1] for node in subgraph]
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)
            padding = 0.1  # Add padding around the subgraph
            bounding_boxes.append([xmin - padding, ymin - padding, xmax + padding, ymax + padding])

        # Draw the graph
        # nx.draw(G, with_labels=True)
        nx.draw(G, lay, labels=self.featNames,with_labels=True, node_size=150, font_size=10,alpha=0.8, font_family='Times New Roman + SimSun')
        # nx.draw_networkx_labels(T, pos, labels=self.featNames, font_family=font_path)  # Draw labels
        weights = {k:round(nx.get_edge_attributes(G, 'weight')[k],1) for k,v in nx.get_edge_attributes(G, 'weight').items()}
        nx.draw_networkx_edge_labels(G,lay,edge_labels=weights, font_size=5)

        # Draw bounding lines for each subgraph
        for i, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
            ls = linestyles[i % len(linestyles)]  # Cycle through linestyles
            c = colors[i % len(colors)]  # Cycle through colors
            plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], ls=ls, color=c, linewidth=2)

        plt.show()

    def draw_multiple_subgraphs(self, subgraphs, subtitles):
        num_subgraphs = len(subgraphs)
        fig, axes = plt.subplots(num_subgraphs, 1, figsize=(4*num_subgraphs, 15))
        
        for i, (subgraph, subtitle, ax) in enumerate(zip(subgraphs, subtitles, axes)):
            # Draw the subgraph
            nx.draw(subgraph, ax=ax, with_labels=True)

            # Add subtitle
            ax.set_title(subtitle, fontsize=12)

        plt.show()

    def visualizeSM(self, T, hit=0, acc=1.0, acc_GT=0.0, deg_std=0.0, showIns=False, savePath=None):
        '''
        To visualize a semantic map with instances on it.
        '''
        font_path = 'times_with_simsun.ttf'
        chinese_font = fm.FontProperties(fname=font_path)
        font_names = ['Noto Sans Mandaic', 'Arial', 'Comic Sans MS', ]
        plt.rcParams['font.family'] = [ 'Heiti TC','Arial', 'Helvetica', 'Times New Roman']
        fig = plt.figure()#figsize=(100,100))
        if not acc_GT is None:
            plt.title("Semantic Map with weights %.2f; %d hit; %.2f ACC from connectivity;\n %.2f ACC_GT; Std degree: %.2f"%(T.size(weight="weight"), hit,acc,acc_GT, deg_std))
        else:
            plt.title("Semantic Map with weights %.2f; %d hit; %.2f ACC from connectivity;\n STd degree: %.2f"%(T.size(weight="weight"), hit,acc,deg_std))
        pos = nx.planar_layout(T)
        nx.draw(T, pos, labels=self.featNames,with_labels=True, node_size=400, font_size=10,alpha=0.8, font_family='Times New Roman + SimSun')
        # nx.draw_networkx_labels(T, pos, labels=self.featNames, font_family=font_path)  # Draw labels
        weights = {k:round(nx.get_edge_attributes(T, 'weight')[k],1) for k,v in nx.get_edge_attributes(T, 'weight').items()}
        edge_width = [T[u][v]['weight'] * 0.7 for u, v in T.edges()]
        nx.draw_networkx_edges(T, pos, width=edge_width, edge_color="plum")
        nx.draw_networkx_edge_labels(T,pos,edge_labels=weights, font_size=5)

        if savePath:
            plt.savefig(savePath, dpi=300)
        plt.show()

        if showIns:
            subgraphs = self.subG_list
            num_subgraphs = len(self.subG_list)
            subtitles = [str(i) for i in range(num_subgraphs)]
            fig, axes = plt.subplots(num_subgraphs//2, 2, figsize=(num_subgraphs*3, 5))
            axes = [aa for a in axes for aa in a]
            
            for i, (subgraph, subtitle, ax) in enumerate(zip(subgraphs, subtitles, axes)):
                # nx.draw(subgraph, ax=ax, with_labels=True)
                print(ax)
                nx.draw(subgraph, pos, ax=ax,labels=self.featNames,with_labels=True, node_size=100, font_size=7,alpha=0.8, font_family='Times New Roman + SimSun')
                ax.set_title(subtitle, fontsize=6)
                weights = {k:nx.get_edge_attributes(subgraph, 'weight')[k] for k,v in nx.get_edge_attributes(subgraph, 'weight').items()}
                nx.draw_networkx_edge_labels(subgraph,pos,ax=ax,edge_labels=weights, font_size=5)

            plt.show()

    def visualizeSM_2D(self):
        self.constG()
        # Perform MDS to embed features into 2D space
        mds = MDS(n_components=2)
        feature_positions = mds.fit_transform(self.tfM.T)

        # Plot the features as points
        plt.figure(figsize=(8, 6))
        plt.scatter(feature_positions[:, 0], feature_positions[:, 1])

        # Label the points (optional)
        font_path = 'times_with_simsun.ttf'
        chinese_font = fm.FontProperties(fname=font_path)
        feature_names = [v for k,v in self.featNames.items()]
        annotations = []
        for i, name in enumerate(feature_names):
            annotations.append(plt.text(feature_positions[i, 0], feature_positions[i, 1], name,font_properties=chinese_font))
        # adjustText.adjust_text(annotations)#,arrowprops=dict(arrowstyle='->', color='red'))   

        plt.title("Feature Similarity Visualization using MDS")
        plt.show()

    def get_optimal_SpanningTrees(self, acc_thr = 1.0, figPath=None):
        '''
        To get all the spanning trees given a graph. The trees should be ordered according to its sum of weights.
        '''
        self.constG()
        # sys.exit(0)
        trees = nx.algorithms.tree.mst.SpanningTreeIterator(self.G, minimum=False)
        selected_ins = range(len(self.tfM))#[0,1,8,-1,-6]##[0,1]
        optimal_trees = []
        for ix, t in enumerate(trees):
            if ix % 1 == 0:
                print(f"This is the id of the spanning tree: {ix}")
                flag, acc, deg_m, deg_std = self.check_subGraph_connectivity(t, selected_ins)
                print(ix, flag)
                print("ACC: %.2f; Weights: %.2f; Mean degree: %.2f; Std degree: %.2f" % (acc, t.size(weight="weight"), deg_m, deg_std))
                if acc >= acc_thr:
                    optimal_trees.append(t)
                    if not self.GT_adj is None:
                        acc_GT = self.accWithGT(t)
                    else:
                        acc_GT = None
                    if acc_GT:
                        print("ACC_GT: %.2f"%acc_GT)
                    # self.highlight_subgraphs(t, subG_list)
                    self.wrongcases = [ix for ix,i in enumerate(flag) if not i]
                    print("Wrong instances id: %s" % self.wrongcases)
                    self.visualizeSM(t, hit=ix, acc=acc, acc_GT=acc_GT, deg_std=deg_std, showIns=False, savePath=figPath[:-4]+f"_{ix}"+figPath[-4:])
                else:
                    print(f"End in iter: {ix} because it has lower acc before maximum iteration.")
                    break
                if ix == 10 or acc < acc_thr:
                    print(f"End because it reaches maximum iteration {ix}")
                    break
                # self.visualizeSM(t)

        print("There are %d optimal trees" % len(optimal_trees))
        # _ = [self.visualizeSM(t) for t in optimal_trees]
