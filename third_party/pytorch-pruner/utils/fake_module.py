import torch
import functools
from torch.nn.modules.loss import _Loss


class FakeModule(torch.nn.Module):
    def __init__(self):
        super(FakeModule, self).__init__()

    def bind_grad_fn(self, result):
        var = result
        while not isinstance(var, torch.Tensor):
            if isinstance(var, dict):
                var = next((v for v in var.values() if isinstance(v[0], torch.Tensor)))
            else:
                var = var[0]
        # we use list here to deal with networks with shared conv
        if hasattr(self, 'grad_fn') and not issubclass(type(self), _Loss):
            self.grad_fn.append(var.grad_fn)
        else:
            self.grad_fn = [var.grad_fn]

    def remove_hooks(self):
        for hook in self.hook_handles:
            hook.remove()
        self.hook_handles.clear()
        if hasattr(self, 'hooks_fb'):
            for hook in self.hooks_fb:
                hook.remove()
            self.hooks_fb.clear()

    def __call__(self, *input, **kwargs):
        '''
            copy & paste from torch.nn.Module.__call__ with additional line to bind grad_fn
        '''
        for hook in self._forward_pre_hooks.values():
            hook(self, input)
        result = self.forward(*input, **kwargs)
        self.bind_grad_fn(result)

        for hook in self._forward_hooks.values():
            hook_result = hook(self, input, result)
            if hook_result is not None:
                raise RuntimeError(
                    "forward hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))
        if len(self._backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                else:
                    var = var[0]
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in self._backward_hooks.values():
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    # register backward hook will register hook for tensor here
                    # remove backward hook can't remove these hooks, so we collect hooks here manually
                    self.hook_handles.append(grad_fn.register_hook(wrapper))
        return result
