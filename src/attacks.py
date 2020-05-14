import torch.nn as nn
import torch
import math
from utils import extract_class_indices

class FastGradientMethod():
    def __init__(self,
                 norm='inf',
                 epsilon=0.3,
                 attack_mode='context',
                 class_fraction=0.2,
                 shot_fraction=0.2,
                 clip_max=1.0,
                 clip_min=-1.0):
        self.norm = norm
        self.epsilon = epsilon
        self.attack_mode = attack_mode
        self.class_fraction = class_fraction
        self.shot_fraction = shot_fraction
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.loss = nn.CrossEntropyLoss()

    def generate(self, context_images, context_labels, target_images, model):
        adv_context_indices = self._generate_context_attack_indices(context_labels)
        context_images.requires_grad = True
        logits = model(context_images, context_labels, target_images)
        labels = self._convert_labels_(logits[0])

        # compute loss
        loss = self.loss(logits[0], labels)
        model.zero_grad()

        # compute gradients
        loss.backward()
        grad = context_images.grad

        # apply norm bound
        if self.norm == 'inf':
            perturbation = torch.sign(grad)

        adv_context_images = context_images.clone()
        for index in adv_context_indices:
            adv_context_images[index] = torch.clamp(adv_context_images[index] +
                                                    self.epsilon * perturbation[index], self.clip_min, self.clip_max)

        return adv_context_images, adv_context_indices

    # Potentially cache class distribution later
    def _generate_context_attack_indices(self, class_labels,):
        indices = []
        classes = torch.unique(class_labels)
        num_classes = len(classes)
        num_classes_to_attack = max(1, math.ceil(self.class_fraction * num_classes))
        for c in range(num_classes_to_attack):
            shot_indices = extract_class_indices(class_labels, c)
            num_shots_in_class = len(shot_indices)
            num_shots_to_attack = max(1, math.ceil(self.shot_fraction * num_shots_in_class))
            attack_indices = shot_indices[0:num_shots_to_attack]
            for index in attack_indices:
                indices.append(index)
        return indices

    @staticmethod
    def _convert_labels_(predictions):
        return torch.argmax(predictions, dim=1, keepdim=False)
