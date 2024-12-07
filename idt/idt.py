import re
import matplotlib
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import f1_score
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Batch



# IDT CLASS ===================================================================================================================
class IDT:
    def __init__(self, width=10, sample_size=20, layer_depth=3, max_depth=5, ccp_alpha=.0):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        #NEW("STARTING :: idt.py -> class IDT -> __init__()")
        import logging
        logging.getLogger('lightning').setLevel(0)
        self.width = width
        self.sample_size = sample_size
        self.layer_depth = layer_depth
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.layer = None
        self.out_layer = None
        #END("COMPLETED :: idt.py -> class IDT -> __init__()")


    
    def fit(self, batch, values, y, sample_weight=None):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        NEW("STARTING :: idt.py -> class IDT -> fit()")
        if self.sample_size is None or self.sample_size > len(batch):
            #NXT("correcting sample size: sample_size = " + str(self.sample_size) + "but batch size = " + str(len(batch)))
            self.sample_size = len(batch)
        #init
        NUM_LAYERS = 0
        self.layer = []
        adj = to_scipy_sparse_matrix(batch.edge_index, num_nodes=batch.x.shape[0]).tobsr()
        x = batch.x.numpy()
        node_sample_weight = sample_weight[batch.batch] if sample_weight is not None else None
        depth_indices = [x.shape[1]]
        # iterate over GNN weights (corresponds to LearnIDTLayer() in article)
        for i, value in enumerate(values):
            new_layers = []
            depth_indices_new = []
            # iterate over width
            for _ in range(self.width):
                NUM_LAYERS += 1
                new_layers.append(IDTInnerLayer(self.layer_depth, depth_indices))
                samples = np.random.choice(np.arange(len(batch)), size=min(self.sample_size, x.shape[0]), replace=False)
                small_batch = Batch.from_data_list(batch[samples])
                small_batch_indices = torch.arange(len(batch.batch))[(batch.batch == torch.tensor(samples).view(-1, 1)).max(axis=0).values]
                new_layers[-1].fit(
                    x[small_batch_indices],
                    value[small_batch_indices],
                    to_scipy_sparse_matrix(small_batch.edge_index, num_nodes=small_batch_indices.shape[0]).tobsr(),
                    node_sample_weight[small_batch_indices] if node_sample_weight is not None else None
                )
                depth_indices_new.append(2 ** (self.layer_depth + 1) - 1)
            x_new = np.concatenate([layer.predict(x, adj) for layer in new_layers], axis=1)
            x = np.concatenate([x, x_new], axis=1)
            self.layer += new_layers
            depth_indices += depth_indices_new
        NXT(str(NUM_LAYERS) + " layers computed")
        self.out_layer = IDTFinalLayer(
            self.max_depth,
            self.ccp_alpha,
            depth_indices
        )
        NXT("out_layer computed")
        #NXT("computing y")
        y = self.out_layer.fit(x, batch.batch, y)
        #END("COMPLETED :: idt.py -> class IDT -> fit()")
        return self


    
    def predict(self, batch):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        x = batch.x.numpy()
        adj = to_scipy_sparse_matrix(batch.edge_index, num_nodes=x.shape[0]).tobsr()
        for i in range(len(self.layer) // self.width):
            x_new = np.concatenate([self.layer[j].predict(x, adj) for j in range(i * self.width, (i + 1) * self.width)], axis=1)
            x = np.concatenate([x, x_new], axis=1)
        return self.out_layer.predict(x, batch.batch)


    
    def accuracy(self, batch):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        prediction = self.predict(batch)
        return (prediction == batch.y.numpy()).mean()


    
    def prune(self):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        NEW("STARTING :: idt.py -> class IDT -> prune()")
        relevant = self.out_layer._relevant()
        for i, layer in enumerate(self.layer[::-1]):
            _relevant = {index for (depth, index) in relevant if depth == len(self.layer) - i - 1}
            layer._prune_irrelevant(_relevant)
            layer._remove_redunant()
            relevant |= layer._relevant()
        #NXT(str(len(relevant)) + " layers remaining")
        #NXT(str(relevant))
        #NXT(str(len(self.layer)) + " layers remaining")
        #HLT(str(dir(self.layer)[0]))
        #HLT(str(self.layer))
        j=0
        for l in self.layer:
            #from pprint import pprint
            #print(j)
            #pprint(vars(l))
            #print("")
            j += 1
        #HLT("leaf_formulas :: " + str(self.layer.leaf_formulas))
        #HLT("dt            :: " + str(self.layer.dt))
        
        #HLN()
        #MRK(str(self.out_layer))
        #self.out_layer
        #HLN()
        END("COMPLETED :: idt.py -> class IDT -> prune()")
        return self


        
    def save_image(self, path):
        import matplotlib.pyplot as plt
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        fig, ax = plt.subplots(
            ncols=self.width, nrows=len(self.layer)//self.width + 1,
            figsize=(4 * self.width, 4 * (len(self.layer)//self.width + 1))
        )
        for i, layer in enumerate(self.layer):
            layer.plot(ax[i // self.width, i % self.width], i)
        self.out_layer.plot(ax[-1, 0], len(self.layer))
        for i in range(1, self.width):
            plt.delaxes(ax[-1, i])
        fig.savefig(path)
        plt.close(fig)


    
    #def plot(self):
    #    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        #NEW("STARTING :: idt.py -> class IDT -> plot()")
    #    import matplotlib.pyplot as plt
    #    fig, ax = plt.subplots(
    #        figsize=(8, 8)
    #    )
    #    self.out_layer.plot(ax, len(self.layer))
    #    ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]
        #END("COMPLETED :: idt.py -> class IDT -> plot()")
    #    plt.show()



    def plot(self):
        import matplotlib.pyplot as plt
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        NEW("STARTING :: idt.py -> class IDT -> plot()")
        j = 0
        for i, layer in enumerate(self.layer):
            fig, axs = plt.subplots(#ncols=self.width, nrows=len(self.layer)//self.width + 1,
                figsize=(16,16))
            PLT("plotting layer " + str(i))
            try:
                #plot = layer.plot(axs[i // self.width, i % self.width], i)
                plot = layer.plot(axs,i)
                if plot == 0:
                    END("nothing to plot")
                    plt.close()
                    continue
                END("successfully plotted layer " + str(i))
                j += 1
                plt.close()
            except:
                ERR("failed to plot layer " + str(i))
                plt.close()
                continue
        fig, ax = plt.subplots(figsize=(16, 16))
        PLT("plotting out layer")
        self.out_layer.plot(ax, len(self.layer))
        ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]
        print("")
        NXT("number of (unpruned) layers = " + str(j))
        END("COMPLETED :: idt.py -> class IDT -> plot()")
        plt.show()
        plt.close()


    
    def fidelity(self, batch, model):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        prediction = self.predict(batch)
        with torch.no_grad():
            model_pred = model(batch).argmax(dim=1).numpy()
        return (prediction == model_pred).mean()


    
    def f1_score(self, batch, average='macro'):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        prediction = self.predict(batch)
        return f1_score(batch.y.numpy(), prediction, average=average)



# IDT INNER LAYER CLASS =======================================================================================================
class IDTInnerLayer:
    def __init__(self, max_depth, depth_indices):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        self.max_depth = max_depth
        self.depth_indices = [index for index in depth_indices]
        self.n_features_in = sum(depth_indices)
        self.dt = None
        self.leaf_indices = None
        self.leaf_values = None
        self.leaf_formulas = None


    
    def fit(self, x, y, adj, sample_weight=None):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        if self.n_features_in == 0:
            x = np.ones((x.shape[0], 1))

        x_neigh_out = adj @ x
        deg_out = adj.sum(axis=1) # out-neighbors are handled as before

        x_neigh_in = adj.T @ x
        deg_in = adj.T.sum(axis=1) # transpose represents in-neighbors

        x = np.asarray(np.concatenate([
            x,
            x_neigh_out,
            x_neigh_in,
            x_neigh_out / deg_out.clip(1e-6, None),
            x_neigh_in / deg_in.clip(1e-6, None)
        ], axis=1))
        self.dt = DecisionTreeRegressor(max_depth=self.max_depth, splitter='random')
        self.dt.fit(x, y, sample_weight=sample_weight)
        leaves = _leaves(self.dt.tree_)
        leaf_values = [self.dt.tree_.value[i, :, 0] for i in leaves]
        if len(leaves) == 1:
            combinations = [{0}]
        else:
            clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
            clustering.fit(leaf_values)
            combinations = _agglomerate_labels(len(leaves), clustering)
        self.leaf_indices = np.array([
            leaves.index(i) if i in leaves else -1
            for i in range(self.dt.tree_.node_count)
        ])
        self.leaf_values = np.array([
            [i in combination for combination in combinations]
            for i in self.leaf_indices if i != -1
        ])
        self.leaf_formulas = [
            [i for (i, b) in enumerate(self.leaf_values[j]) if b]
            for j in range(len(self.leaf_values))
        ]
        #HLT(str(self.leaf_formulas))
        return self


    
    def predict(self, x, adj=None):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        if self.n_features_in == 0:
            x = np.ones((x.shape[0], 1))
        x_neigh_out = adj @ x
        deg_out = adj.sum(axis=1) # out-neighbors are handled as before

        x_neigh_in = adj.T @ x
        deg_in = adj.T.sum(axis=1) # transpose represents in-neighbors

        x = np.asarray(np.concatenate([
            x,
            x_neigh_out,
            x_neigh_in,
            x_neigh_out / deg_out.clip(1e-6, None),
            x_neigh_in / deg_in.clip(1e-6, None)
        ], axis=1))
        pred = self.dt.apply(x)
        return self.leaf_values[self.leaf_indices[pred]]


    
    def fit_predict(self, x, y, adj=None):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        self.fit(x, y, adj)
        return self.predict(x, adj)


    
    def _relevant(self):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        return {_feature_depth_index(feature, self.depth_indices) for feature in self.dt.tree_.feature if feature != -2}


        
    def _prune_irrelevant(self, relevant):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        self.leaf_formulas = [
            [i for i in formulas if i in relevant]
            for formulas in self.leaf_formulas
        ]


    
    def _remove_redunant(self):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        flag = True
        while flag:
            flag = False
            for parent, (left, right) in enumerate(zip(self.dt.tree_.children_left, self.dt.tree_.children_right)):
                if self.leaf_indices[left] != -1 and self.leaf_indices[right] != -1 and left != -1 and right != -1 and \
                        self.leaf_formulas[self.leaf_indices[left]] == self.leaf_formulas[self.leaf_indices[right]]:
                    self._merge_leaves(parent, left, right)
                    flag = True
                    break


    
    def _merge_leaves(self, parent, left, right):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        self.dt.tree_.children_left[parent] = -1
        self.dt.tree_.children_right[parent] = -1
        self.leaf_indices[parent] = self.leaf_indices[left]
        self.leaf_indices[left] = -1
        self.leaf_indices[right] = -1
        self.dt.tree_.feature[parent] = -2
        self.dt.tree_.threshold[parent] = -2


    '''def plot(self, ax, n=0):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        from sklearn.tree import plot_tree
        plot_tree(self.dt, ax=ax)'''

    
    #def plot(self):
    def plot(self, ax, n=0):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        n_nodes = self.dt.tree_.node_count
        children_left = self.dt.tree_.children_left
        children_right = self.dt.tree_.children_right
        feature = self.dt.tree_.feature
        threshold = self.dt.tree_.threshold
        values = self.dt.tree_.value
        
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth = stack.pop()
            node_depth[node_id] = depth
        
            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            if is_split_node:
                stack.append((children_left[node_id], depth + 1))
                stack.append((children_right[node_id], depth + 1))
            else:
                is_leaves[node_id] = True

        prune = True
        prune_nodes = 0
        for i in range(n_nodes):
            if is_leaves[i] == False:
                if threshold[i] != -2:
                    prune = False
                else:
                    prune_nodes += 1
        if prune == True:    
            NXT("layer pruned")
            return 0
        
        print(
            "The binary tree structure has {n} nodes and has "
            "the following tree structure:\n".format(n=n_nodes-prune_nodes)
        )
        for i in range(n_nodes):
            if is_leaves[i]:
                print(
                    "{space}node={node} is a leaf node with value={value}.".format(
                        space=node_depth[i] * "\t", node=i, value=len(values[i])  #np.around(values[i], 3)
                    )
                )
            else:
                if threshold[i] != -2:
                    print(
                        "{space}node={node} is a split node with value={value}: "
                        "go to node {left} if X[:, {feature}] <= {threshold} "
                        "else to node {right}.".format(
                            space=node_depth[i] * "\t",
                            node=i,
                            left=children_left[i],
                            feature=feature[i],
                            threshold=threshold[i],
                            right=children_right[i],
                            value=len(values[i])  #np.around(values[i], 3),
                        )
                    )
        '''
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        from sklearn.tree import plot_tree
        plot_tree(self.dt, ax=ax)
        leaf_counter = 0
        #HLT("looping over obj in ax.properties()['children']")
        txt = ""
        log = False
        for obj in ax.properties()['children']:
            #HLT("loop " + str(leaf_counter))
            if type(obj) == matplotlib.text.Annotation:
                #CND("type(obj)", "matplotlib.text.Annotation", type(obj), matplotlib.text.Annotation, "==", print_RHS=False)
                obj.set_fontsize(8)
                txt = obj.get_text().splitlines()[0]
                match = re.match(r'x\[(\d+)\] <= (\d+\.\d+)', txt)
                skip = False
                if match:
                    log = True
                    NXT("MATCH      :: txt = " + str(txt))
                    #CND("txt", r'x\[(\d+)\] <= (\d+\.\d+)', txt, r'x\[(\d+)\] <= (\d+\.\d+)', "regex match", print_RHS=False)
                    feature, threshold = match.groups()
                    feature = int(feature)
                    formula = _feature_formula(feature, self.depth_indices)
                    threshold = float(threshold)
                    if feature < self.n_features_in:
                        #CND("feature", "self.n_features_in", feature, self.n_features_in, "<")
                        #NXT("\tfeature = " + str(feature) + " < " + str(self.n_features_in) + " = self.n_features_in")
                        obj.set_text(fr'$I{formula} > 0$')
                    elif feature < 2 * self.n_features_in:
                        #CND("feature", "2 * self.n_features_in", feature, 2 * self.n_features_in, "<")
                        #NXT("\tfeature = " + str(feature) + " < " + str(2 * self.n_features_in) + " = 2*self.n_features_in")
                        obj.set_text(fr'$A{formula} > {int(threshold)}$')
                    elif feature < 3 * self.n_features_in:
                        #CND("feature", "3 * self.n_features_in", feature, 3 * self.n_features_in, "<")
                        #NXT("\tfeature = " + str(feature) + " < " + str(3*self.n_features_in) + " = 3*self.n_features_in")
                        obj.set_text(fr'$A^T{formula} > {int(threshold)}$')
                    elif feature < 4 * self.n_features_in:
                        #CND("feature", "4 * self.n_features_in", feature, 4 * self.n_features_in, "<")
                        #NXT("\tfeature = " + str(feature) + " < " + str(4*self.n_features_in) + " = 4*self.n_features_in")
                        obj.set_text(fr'$A{formula} > {threshold}$')
                    elif feature < 5 * self.n_features_in:
                        #CND("feature", "5 * self.n_features_in", feature, 5 * self.n_features_in, "<")
                        #NXT("\tfeature = " + str(feature) + " < " + str(5*self.n_features_in) + " = 5*self.n_features_in")
                        obj.set_text(fr'$A^T{formula} > {threshold}$')
                    NXT("formula = " + str(formula))
                else:
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    #CND("txt", r'x\[(\d+)\] <= (\d+\.\d+)', txt, r'x\[(\d+)\] <= (\d+\.\d+)', "regex ! match", print_RHS=False)
                    txt = r"$" + r", "
                    if log:
                        HLT("GENERATING :: txt (leaf_counter = " + str(leaf_counter) + ")")
                    index_list = []


                    if log:
                        NXT("computing index list :: " + str(len(self.leaf_indices)) + " :: " + str(self.leaf_indices))
                    for i in self.leaf_indices:
                        NXT("\ti = " + str(i))
                        if i != -1:
                            if log:
                                NXT("\tnext element = " + str(i))
                            index_list.append(i)
                    DBG("finished iterating leaf_indices",1)
                    
                    if len(index_list) <= leaf_counter:
                        DBG("index_list <= leaf_counter")
                        return 1        
                    
                    if log:
                        #END("successfully computed index list = " + str(index_list))
                        NXT("index list = " + str(index_list))
                    
                    #if log:
                        #NXT("computing index (element " + str(leaf_counter) + " of index_list)")
                    if log:
                        NXT("index = " + str(index_list[leaf_counter]))
                    
                    #if log:
                        #NEW("joining leaf_formulas to txt")
                    for i in self.leaf_formulas[index_list[leaf_counter]]:
                        if log:
                            NXT("forumla = " + fr"M_{{{i}}}^{{{n}}}")
                        txt.join(fr"M_{{{i}}}^{{{n}}}")
                    
                    txt = txt + r"\:$"
                    if log == False:
                        NXT("GENERATED :: txt = " + str(txt))
                    
                    #if log:
                        #END("joined leaf_formulas to txt = " + str(txt))
                    
                    #txt = r"$" + r", ".join([
                    #    fr"M_{{{i}}}^{{{n}}}" for i in self.leaf_formulas[[i for i in self.leaf_indices if i != -1][leaf_counter]]
                    #]) + r"\:$"
                    obj.set_text(txt)
                    leaf_counter += 1
            #else:
                #NXT("type(obj) = " + str(type(obj)))
        if txt == "$, \:$":
            return 0
        else:
            HLT("txt = " + str(txt))
            return 1
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''


# IDT FINAL LAYER =============================================================================================================
class IDTFinalLayer:
    def __init__(self, max_depth, ccp_alpha, depth_indices):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.depth_indices = [index for index in depth_indices]
        self.n_features_in = sum(depth_indices)
        self.dt = None
        self.leaf_indices = None
        self.leaf_values = None


    
    def fit(self, x, batch, y, sample_weight=None):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        x = _pool(x, batch)
        self.dt = DecisionTreeClassifier(max_depth=self.max_depth, ccp_alpha=self.ccp_alpha)
        self.dt.fit(x, y, sample_weight=sample_weight)


    
    def predict(self, x, batch):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        x = _pool(x, batch)
        return self.dt.predict(x)


    
    def fit_predict(self, x, batch, y):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        self.fit(x, batch, y)
        return self.predict(x, batch)


    
    def _relevant(self):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        return {_feature_depth_index(feature, self.depth_indices) for feature in self.dt.tree_.feature if feature != -2}


        
    def plot(self, ax, n=0):
        from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
        from sklearn.tree import plot_tree
        plot_tree(self.dt, ax=ax)
        for obj in ax.properties()['children']:
            if type(obj) == matplotlib.text.Annotation:
                obj.set_fontsize(12)
                txt = obj.get_text().splitlines()[0]
                match = re.match(r'x\[(\d+)\] <= (\d+\.\d+)', txt)
                if match:
                    feature, threshold = match.groups()
                    feature = int(feature)
                    formula = _feature_formula(feature, self.depth_indices)
                    threshold = float(threshold)
                    if feature < self.n_features_in:
                        obj.set_text(fr'$1{formula} > {threshold}$')
                    else:
                        obj.set_text(fr'$1{formula} > {int(threshold)}$')
                if 'True' in txt:
                    obj.set_text(txt.replace('True', 'False'))
                elif 'False' in txt:
                    obj.set_text(txt.replace('False', 'True'))



# UTILITY FUNCTIONS ===========================================================================================================



def replace_text(obj):
    if type(obj) == matplotlib.text.Annotation:
        txt = obj.get_text()
        txt = re.sub("gini[^$].*\n","",txt)
        txt = re.sub("\nvalue[^$].*","",txt)
        txt = re.sub("samples = ","",txt)
        obj.set_text(txt)
    return obj

    

def _pool(x, batch):
    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
    return np.concatenate([
        global_mean_pool(torch.tensor(x), batch).numpy(),
        global_add_pool(torch.tensor(x), batch).numpy()
    ], axis=1)



def _agglomerate_labels(n_labels, clustering):
    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
    agglomerated_features = [{i} for i in range(n_labels)]
    for i, j in clustering.children_:
        agglomerated_features.append(agglomerated_features[i] | agglomerated_features[j])
    return agglomerated_features



def _leaves(tree, node_id=0):
    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
    if tree.children_left[node_id] == tree.children_right[node_id] and \
        (node_id in tree.children_left or node_id in tree.children_right or node_id == 0):
        return [node_id]
    else:
        left = _leaves(tree, tree.children_left[node_id])
        right = _leaves(tree, tree.children_right[node_id])
        return left + right

        

def get_activations(batch, model_or_int):
    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
    if isinstance(model_or_int, int):
        return _get_values_int(batch, model_or_int)
    if isinstance(model_or_int, torch.nn.Module):
        return _get_values_model(batch, model_or_int)
    raise ValueError(f"Unknown type {type(model_or_int)}")



def _get_values_int(batch, int):
    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
    labels = batch.y.numpy().max() + 1
    values = []
    for datapoint in batch.to_data_list():
        onehot = np.eye(labels)[datapoint.y.numpy()]
        values.append(onehot.repeat(datapoint.x.shape[0], axis=0))
    values = np.concatenate(values, axis=0).squeeze()
    values = [values] * int
    return values



def _get_values_model(batch, model):
    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
    values = []
    hook = model.act.register_forward_hook(
        lambda mod, inp, out: values.append(out.detach().numpy().squeeze())
    )
    with torch.no_grad():
        model(batch).detach().numpy()
    hook.remove()
    return values[:-1]



def _feature_formula(index, depth_indices):
    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
    depth, index = _feature_depth_index(index, depth_indices)
    if depth == -1:
        return fr'U_{{{index}}}'
    else:
        return fr'\chi_{{{index}}}^{{{depth}}}'
"""
def _feature_formula(index, depth_indices):
    depth, index = _feature_depth_index(index, depth_indices)
    n_features_in = sum(depth_indices)
    num_feature_types = 5  # Number of feature types per depth
    features_per_type = n_features_in // num_feature_types

    if depth == -1:
        return fr'U_{{{index}}}'
    else:
        # Determine which feature type the index corresponds to
        feature_type = index // features_per_type
        feature_index = index % features_per_type

        if feature_type == 0:
            modal_param = 'I'
        elif feature_type == 1:
            modal_param = 'A'
        elif feature_type == 2:
            modal_param = 'A^T'
        elif feature_type == 3:
            modal_param = r'\frac{A}{d_{\text{out}}}'
        elif feature_type == 4:
            modal_param = r'\frac{A^T}{d_{\text{in}}}'
        else:
            modal_param = 'Unknown'

        formula = fr'\chi_{{{feature_index}}}^{{{depth}}}'
        return fr'{modal_param}{formula}'

"""



def _feature_depth_index(index, depth_indices):
    from idt.ui import DBG, HLT, NEW, NXT, END, MRK, HLN, CND, ERR, PLT
    index = index % sum(depth_indices)
    depth = -1
    for i in depth_indices:
        if index < i:
            return depth, index
        index -= i
        depth += 1
