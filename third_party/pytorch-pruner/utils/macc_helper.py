import numpy as np
import torch


def model_macc(model, input, split=False):
    hook_list = []

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        params = output_channels * kernel_ops
        macc = params * output_height * output_width
        list_conv.append(macc)

    def conv_split_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        kernel_ops = 2 * (self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)+1)
        params = output_channels * kernel_ops
        macc = params * output_height * output_width
        list_conv.append(macc)

    list_linear = []

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement()
        bias_ops = self.bias.nelement()
        macc = (weight_ops + bias_ops)
        list_linear.append(macc)

    def linear_split_hook(self, input, output):
        weight_ops = self.weight.nelement()
        bias_ops = self.bias.nelement()
        macc = (2 * weight_ops + bias_ops)
        list_linear.append(macc)

    def foo(net):
        childrens = list(net.children())
        if split is False:
            chook = conv_hook
            lhook = linear_hook
        else:
            chook = conv_split_hook
            lhook = linear_split_hook

        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                hook_list.append(net.register_forward_hook(chook))
            elif isinstance(net, torch.nn.Linear):
                hook_list.append(net.register_forward_hook(lhook))
            return
        for c in childrens:
            foo(c)

    model = model.cuda()
    foo(model)
    last_state = model.training
    model.eval()
    output = model(input)
    model.train(mode=last_state)
    total_macc = sum(list_conv) + sum(list_linear)
    for hook in hook_list:
        hook.remove()
    return output, total_macc / np.math.pow(10, 9)
