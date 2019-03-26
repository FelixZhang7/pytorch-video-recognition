# coding:utf-8
import copy
import logging
import torch.nn as nn
from .trim_helper import _direct_prunable_type, _bn_type, splitByModel, getAttr

logger = logging.getLogger('PRUNER')


class Trimmer(object):
    '''
        change the topology of graph  during the whole traininig process.
    '''
    def __init__(self, model, graph):
        self.model = model
        self.graph = graph
        self.dparent_list = copy.deepcopy(graph.parent_list)
        self.dchild_list = copy.deepcopy(graph.child_list)
        self.node_list = graph.node_list

    def _topology(self):
        '''
            return the topology info of model
        '''
        assert self.graph is not None
        return self.graph.node_list, self.graph.parent_list, self.graph.child_list

    def _build_conv_graph(self):
        '''
            make conv layer's dparent_list and dchild_list only contains valid info
            excluding: ReLU, Pooling ... nolinear info)
        '''
        node_list = self.node_list
        graph = self.graph

        def isnolinear(typename, aux):
            return typename in ['ReLU', 'Softmax', 'Pooling', 'Dropout', 'ReLU6',
                                'Sigmoid', 'BatchNorm', 'Concat', 'PSROIPooling', 'Interp'] + aux

        def backtrace(grad_fn, glist, aux=[]):
            n_id = node_list.index(grad_fn)
            typename = self.graph.get_pytorch_type(grad_fn)
            if not isnolinear(typename, aux) or len(glist[n_id]) == 0:
                return [n_id]
            trace_list = []
            for g in glist[n_id]:
                t_list = backtrace(node_list[g], glist, aux=aux)
                for t in t_list:
                    trace_list.append(t)
            return trace_list

        def get_dlist(node_id, _list, aux=[]):
            d_list = []
            for node in _list[node_id]:
                t_list = backtrace(node_list[node], _list, aux=aux)
                for t in t_list:
                    d_list.append(t)
            return d_list

        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type):
                for grad_fn in module.grad_fn:
                    node_id = node_list.index(grad_fn)
                    self.dparent_list[node_id] = get_dlist(node_id, graph.parent_list)
                    self.dchild_list[node_id] = get_dlist(node_id, graph.child_list)

    def _get_bnlist(self, nid):
        '''
            make bn layer's dparent_list and dchild_list only contains conv layer
        '''
        node_list, parent_list, child_list = self._topology()
        convitems = []
        listp = parent_list[nid]

        for p in listp:
            tnode = self.node_list[p]
            typename = self.graph.get_pytorch_type(tnode)
            if (typename not in ['Convolution', 'Concat']):
                if typename == 'Eltwise':
                    parent = parent_list[p][0]
                    if self.graph.get_pytorch_type(node_list[parent]) == 'Convolution':
                        items = [parent]
                    else:
                        items = self._get_bnlist(parent)
                else:
                    items = self._get_bnlist(p)
                for item in items:
                    convitems.append(item)
            else:
                convitems.append(p)
        return convitems

    def _build_bn_graph(self):
        '''
            rebuild bn graph to accelerate the index bn's parent and bn's child info
        '''
        node_list, parent_list, child_list = self._topology()

        for idx, node in enumerate(node_list):
            if self.graph.is_batchnorm(node):
                self.dparent_list[idx] = self._get_bnlist(idx)

    def _get_attrinfo(self, name):
        '''
            derive the module according name
            return the parent instance of this module and the parent instance's name of this module
        '''
        lastname = None
        if '.' in name:
            names = splitByModel(self.model, name.split('.'))
            fmodule = getAttr(self.model, names[:-1])
            lastname = names[-1]
        else:
            fmodule = self.model
            lastname = name
        return lastname, fmodule

    def _get_module_by_node(self, node):
        for module in self.model.modules():
            if hasattr(module, 'grad_fn') and id(node) in list(map(id, module.grad_fn)) and len(module._modules) == 0:
                return module
        return None

    def _get_name_by_module(self, module):
        for name, imodule in self.model.named_modules():
            if id(module) == id(imodule):
                return name
        return None

    def _rebuild_conv(self, node, nid, index, modified, inherit=True):
        '''
            core method of Trimmer class
            rebuild conv layer and its related layer
        '''
        node_list, parent_list, child_list = self._topology()
        # modify self-conv layer
        if self.graph.get_pytorch_type(node) in ['Convolution', 'InnerProduct'] or self.graph.is_batchnorm(node):
            module = self._get_module_by_node(node)
            name = self._get_name_by_module(module)
            lastname, fmodule = self._get_attrinfo(name)
        if self.graph.get_pytorch_type(node) in ['Convolution', 'InnerProduct']:
            if (inherit and self.visdown[name]) or (not inherit and self.visup[name]):
                if name in modified.keys():
                    modified[name][inherit].extend(index)
                else:
                    modified[name] = {True: [], False: []}
                    modified[name][inherit].extend(index)
                if inherit:
                    self.visdown[name] = False
                else:
                    self.visup[name] = False
        else:
            if self.graph.is_batchnorm(node):
                if self.visup[name]:
                    if name in modified.keys():
                        modified[name][False].extend(index)
                    else:
                        modified[name] = {True: [], False: []}
                        modified[name][False].extend(index)
                    self.visup[name] = False
            for chid in self.graph.child_list[nid]:
                if self.graph.get_pytorch_type(self.node_list[chid]) == 'Concat':
                    accumulation = 0
                    # build graph is inverse with forward order
                    for parid in self.graph.parent_list[chid][::-1]:
                        if parid == nid:
                            break
                        else:
                            accumulation += self.lengthmap[self.node_list[parid]]
                    idxp = []
                    for i in range(len(index)):
                        idxp.append(index[i] + accumulation)
                    self._rebuild_conv(self.node_list[chid], chid, idxp, modified, inherit=False)
                else:
                    self._rebuild_conv(self.node_list[chid], chid, index, modified, inherit=False)

        # modify succeed-bn layer
        if inherit and self.graph.get_pytorch_type(node) == 'Convolution':
            module = self._get_module_by_node(node)
            name = self._get_name_by_module(module)
            lastname, fmodule = self._get_attrinfo(name)

            for idx, inode in enumerate(self.node_list):
                if self.graph.is_batchnorm(inode) and nid in self.dparent_list[idx]:
                    oldbn = self._get_module_by_node(inode)
                    oldname = self._get_name_by_module(oldbn)
                    if self.visup[oldname]:
                        lastnameBN, fmoduleBN = self._get_attrinfo(oldname)
                        if oldname in modified.keys():
                            modified[oldname][False].extend(index)
                        else:
                            modified[oldname] = {True: [], False: []}
                            modified[oldname][False].extend(index)
                        self.visup[oldname] = False

            # modify convs in same group
            if module.groupname == self._get_name_by_module(module):
                for iname, imodule in self.model.named_modules():
                    if isinstance(imodule, _direct_prunable_type) and imodule.prunable \
                       and module.groupname == imodule.groupname and id(imodule) != id(module):
                        for grad_fn in imodule.grad_fn:
                            self._rebuild_conv(grad_fn, self.node_list.index(grad_fn), index, modified, inherit=True)

            for chid in self.graph.child_list[nid]:
                if self.graph.get_pytorch_type(self.node_list[chid]) == 'Concat':
                    accumulation = 0
                    # build graph is inverse with forward order
                    for parid in self.graph.parent_list[chid][::-1]:
                        if parid == nid:
                            break
                        else:
                            accumulation += self.lengthmap[self.node_list[parid]]
                    idxp = []
                    for i in range(len(index)):
                        idxp.append(index[i] + accumulation)
                    self._rebuild_conv(self.node_list[chid], chid, idxp, modified, inherit=False)
                else:
                    self._rebuild_conv(self.node_list[chid], chid, index, modified, inherit=False)

    def _findDconv(self, node):
        '''
            this function is related with build_length_map
            find conv layer connected with the node directly as conv layer contains info about channels
        '''
        node_list, parent_list, child_list = self._topology()
        if node in self.lengthmap:
            return self.lengthmap[node]
        module = None
        for imodule in self.model.modules():
            if hasattr(imodule, 'grad_fn') and id(node) in list(map(id, imodule.grad_fn)):
                module = imodule
                break
        if module is not None and isinstance(module, _direct_prunable_type):
            self.lengthmap[node] = module.out_channels
        elif module is not None and isinstance(module, nn.Linear):
            self.lengthmap[node] = module.out_features
        elif module is not None and isinstance(module, _bn_type):
            self.lengthmap[node] = module.num_features
        elif self.graph.get_pytorch_type(node) == 'Concat':
            accumulation = 0
            for nextid in self.graph.parent_list[node_list.index(node)]:
                accumulation += self._findDconv(self.node_list[nextid])
            self.lengthmap[node] = accumulation
        else:
            if len(parent_list[node_list.index(node)]) == 0:
                self.lengthmap[node] = 0
            else:
                nextid = parent_list[node_list.index(node)][0]
                self.lengthmap[node] = self._findDconv(node_list[nextid])
        return self.lengthmap[node]

    def _build_length_map(self):
        '''
            only concat operation existing in model should we build length_map
            notice: len(module._modules) == 0 means that the module is atom module e.g relu, conv, bn ...
        '''
        node_list, parent_list, child_list = self._topology()
        execute = False
        self.lengthmap = {}
        for node in self.node_list:
            if self.graph.get_pytorch_type(node) == 'Concat':
                execute = True
        if execute:
            for node in self.node_list:
                if self.graph.get_pytorch_type(node) == 'Concat':
                    accumulation = 0
                    for nextid in self.graph.parent_list[node_list.index(node)]:
                        accumulation += self._findDconv(node_list[nextid])
                    self.lengthmap[node] = accumulation

                elif self.graph.get_pytorch_type(node) == 'Convolution':
                    for module in self.model.modules():
                        if hasattr(module, 'grad_fn') and id(node) in list(map(id, module.grad_fn)) \
                           and len(module._modules) == 0:
                            self.lengthmap[node] = module.out_channels
                            break
                elif self.graph.get_pytorch_type(node) == 'InnerProduct':
                    for module in self.model.modules():
                        if hasattr(module, 'grad_fn') and id(node) in list(map(id, module.grad_fn)) \
                           and len(module._modules) == 0:
                            self.lengthmap[node] = module.out_features
                            break
                elif self.graph.get_pytorch_type(node) in _bn_type:
                    for module in self.model.modules():
                        if hasattr(module, 'grad_fn') and id(node) in list(map(id, module.grad_fn)) \
                           and len(module._modules) == 0:
                            self.lengthmap[node] = module.num_features
                            break
                else:
                    self.lengthmap[node] = self._findDconv(node)

    def _build_vis(self):
        '''
            build vis mark to record whether this conv or bn is modified.(in risk of dead loop)
        '''
        self.visup = {}
        self.visdown = {}
        for name, module in self.model.named_modules():
            if isinstance(module, _direct_prunable_type) or isinstance(module, _bn_type) \
               or isinstance(module, nn.Linear):
                self.visup[name] = True
                self.visdown[name] = True

    def rebuild_graph(self, name, module, index, modified):
        '''
            rebuild model given name, module and index (filters index to be pruned)
            notice: for correctness, this operation only accounts for pruning one layer.
            multi times recall this function for pruning multi layers
        '''
        logger.info('pruning module {0} {1} {2}'.format(name, index, len(index)))
        self._build_vis()
        self._build_length_map()
        self._build_conv_graph()
        self._build_bn_graph()
        for node in module.grad_fn:
            self._rebuild_conv(node, self.node_list.index(node), index, modified)
