import torch.nn as nn
import torch

class FastGradientMethod():
    def __init__(self, norm='inf', epsilon=0.3, attack_mode='default', clip_max=1.0, clip_min=-1.0):
        self.norm = norm
        self.epsilon = epsilon
        self.attack_mode = attack_mode
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.loss = nn.CrossEntropyLoss()


    def generate(self, context_images, context_labels, target_images, model):
        adv_context_mask = self._generate_context_mask_(context_images, context_labels)
        context_images.requires_grad = True
        logits = model(context_images, context_labels, target_images)
        labels = self._convert_labels_(logits[0])

        # Compute loss
        loss = self.loss(logits[0], labels)
        model.zero_grad()

        loss.backward()
        grad = context_images.grad

        # Apply norm bound
        if self.norm == 'inf':
            perturbation = torch.sign(grad)

        adv_context_image = context_images[0].clone()
        adv_context_image = torch.clamp(adv_context_image + self.epsilon * perturbation[0], self.clip_min, self.clip_max)

        return adv_context_image


    # Potentially cache class distribution later
    def _generate_context_mask_(self, context_images, context_labels):
        return [(0,0)]

    def _convert_labels_(self, preds):
        return torch.argmax(preds, dim=1, keepdim=False)
