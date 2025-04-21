from transformers import Trainer
import torch

class ModernBertTrainer(Trainer):
    def __init__(self, *args, class_weights=None, max_class_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if class_weights:
            self.class_weights = torch.tensor(class_weights)
            if max_class_weight:
                self.class_weights = torch.clamp(self.class_weights, max=max_class_weight)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.class_weights is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
                
        # weighted BCE loss to deal with class imbalance
        labels = inputs.get("labels")

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        loss = (loss * self.class_weights.to(loss.device)).mean()
        
        return (loss, outputs) if return_outputs else loss