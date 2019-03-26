# coding:utf-8


# make backward function and op mapping here
# pytorch dynamically generate backward, we use this mapping to build graph
def get_bp_op_map():
    return {'ConvNdBackward': 'Convolution',
            'AddmmBackward': 'InnerProduct',
            'SyncBNFunc': 'BatchNorm',
            'SyncBNFuncBackward': 'BatchNorm',
            'BatchNormBackward': 'BatchNorm',
            'ThresholdBackward': 'ReLU',
            'SigmoidBackward': 'Sigmoid',
            'LogSoftmaxBackward': 'Softmax',
            'SoftmaxBackward': 'Softmax',
            'MaxPool2DBackward': 'Pooling',
            'AvgPool2DBackward': 'Pooling',
            'ViewBackward': 'Reshape',
            "DropoutBackward": "Dropout",
            "HardtanhBackward": "ReLU6",
            'UpsamplingBilinear2dBackward': 'Interp',
            'UpsamplingNearest2dBackward': 'Interp',  # here only for build graph, not for convert model
            'CatBackward': 'Concat',
            'AddBackward': 'Eltwise',
            'MulBackward': 'Eltwise',
            'CmaxBackward': 'Eltwise',
            'PSRoIPoolFunction': 'PSROIPooling',
            'PSRoIPoolFunctionBackward': 'PSROIPooling',
            'RoIAlignFunction': 'RoIAlign',
            'ThAddBackward': 'Eltwise',
            'ThMulBackward': 'Eltwise',
            'ThCmaxBackward': 'Eltwise',
            'MaxPool2DWithIndicesBackward': 'Pooling',
            'AvgPool2DWithIndicesBackward': 'Pooling',
            'CudnnBatchNormBackward': 'BatchNorm',
            'ThAddmmBackward': 'InnerProduct',
            'CudnnConvolutionBackward': 'Convolution',
            'ThnnConvDepthwise2DBackward': 'Convolution',
            # 'ThnnConvDepthwise2DBackward': 'DepthWiseConvolution',
            }


bp_op_map = get_bp_op_map()


class Graph(object):
    '''
        create a dag based on the dynamic forward.
    '''
    def __init__(self):
        pass

    def is_convolution(self, node):
        return self.get_type(node) in ['ConvNdBackward', 'CudnnConvolutionBackward']

    # def is_depthwise(self, node):
    #    return self.get_type(node) in ['ThnnConvDepthwise2DBackward']

    def is_fc(self, node):
        return self.get_type(node) in ['AddmmBackward', 'ThAddmmBackward']

    def is_batchnorm(self, node):
        return self.get_type(node) in ['BatchNormBackward', 'SyncBNFunc',
                                       'CudnnBatchNormBackward', 'SyncBNFuncBackward']

    def get_type(self, node):
        uname = type(node).__name__
        while uname[-1].isdigit():
            uname = uname[:-1]
        return uname

    def get_pytorch_type(self, node):
        return bp_op_map.get(self.get_type(node))

    def module_to_node_idx(self, mod):
        for i in range(len(self.node_list)):
            if mod.grad_fn[0] == self.node_list[i]:
                return i

    def collect_var(self, root):
        node_list = [root]
        type_list = []
        node_set = {root}
        parent_list = [list()]
        child_list = [list()]

        list_index = 0
        not_supported = set()
        while list_index < len(node_list):
            var = node_list[list_index]
            vtype = self.get_type(var)
            if vtype not in bp_op_map:
                # print('{}, {} not in bp_op_map'.format(var, vtype))
                type_list.append(vtype)
            else:
                type_list.append(bp_op_map[vtype])

            for u in var.next_functions:
                utype = self.get_type(u[0])
                if utype not in bp_op_map:
                    if utype not in ["NoneType", "AccumulateGrad"]:
                        # print('not in:', utype)#提示那些层需要添加
                        pass
                    else:
                        continue

                if u[0] not in node_set:
                    node_set.add(u[0])
                    node_list.append(u[0])
                    parent_list.append(list())
                    child_list.append(list())
                parent = node_list.index(u[0])
                parent_list[list_index].append(parent)
                child_list[parent].append(list_index)
            list_index += 1
        assert len(not_supported) == 0, '{} are not supported'.format(not_supported)
        return node_list, parent_list, child_list, type_list, node_set

    def build_grad_graph(self, output):
        '''
            len(node_list) == len(parent_list) == len(child_list) == len(type_list)
            node_list: list of node, with type(node) == xxxBackward
            parent_list: list of list of parent
        '''
        self.node_list, self.parent_list, self.child_list, self.type_list, self.node_set \
            = self.collect_var(output.grad_fn)
