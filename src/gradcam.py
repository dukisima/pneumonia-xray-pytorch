# Here lays the class used to create GradCAM-s from single batch images
import torch

class GradCAM():
    def __init__(self, model, target_layer ):
        self.model = model
        self.target_layer = target_layer
        self.activations = None #[B,C,H,W], B=batch size(1), C=num of channels, H,W=size of feature maps
        self.gradients = None

        #Forward hook: save activations
        self.fwd_handle = target_layer.register_forward_hook(self._save_activations)
        #Backward hook= save gradients
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        # Removes the hooks after one GradCAM to not affect next GradCAM
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    @torch.no_grad()
    def _normalize(self,x,eps=1e-8):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min)/(x_max-x_min+eps)

    def __call__(self, inputs, target_index=None):
        """
        inputs: tensor [B, 3, H, W] on the same device as model
        target_index: int class index (if None -> use predicted class)
        returns: heatmap per image as numpy array [H, W] in [0,1]
        """
        #0. Deletes old gradients
        self.model.zero_grad(set_to_none=True)

        #1.Forward propagation
        outputs = self.model(inputs)

        #If not specified ti takes the model prediction
        if target_index is None:
            target_index = outputs.argmax(dim=1)  # [B]

        #2. Loss
        loss = outputs.gather(1, target_index.view(-1,1)).sum()
        loss.backward()

        # first and only in the batch
        grads = self.gradients[0] # [C, h, w]
        acts = self.activations[0] # [C, h, w]

        weights = grads.mean(dim=(1, 2))  # [C]

        # weighted sum of activations
        cam = torch.zeros_like(acts[0])
        for c, w in enumerate(weights):
            cam += w * acts[c]

        # ReLU & normalize to [0,1]
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = self._normalize(cam)

        return cam, outputs.detach().softmax(dim=1).cpu().numpy()

