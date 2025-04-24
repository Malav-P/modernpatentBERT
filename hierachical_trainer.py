import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer
import re

class HierarchicalLoss(nn.Module):
    def __init__(self, classes, lambda_hier=0.5, weights=(1.0, 0.5, 0.25)):
        super().__init__()
        # classes is expected to be the *sorted* list of class names
        # corresponding to indices 0, 1, 2, ...
        self.classes = classes
        self.lambda_hier = lambda_hier
        self.distance_weights = {
            1: weights[0], # Section mismatch
            2: weights[1], # Class mismatch
            3: weights[2], # Subclass mismatch
            0: 0.0         # Match
        }
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.class_parts = self._precompute_parts()

    def _precompute_parts(self):
        parts = {}
        # Create mapping from index to parsed parts based on the sorted self.classes list
        for i, code in enumerate(self.classes):
            match = re.match(r"([A-Z])([0-9]{2})([A-Z])", code)
            if match:
                # Section, Class (Sec+Num), Subclass (Sec+Num+Letter)
                parts[i] = (match.group(1), match.group(1)+match.group(2), code)
            else:
                # Handle codes that don't fit the pattern (e.g., Y10S) - treat as single level
                parts[i] = (code[0], code[:min(3, len(code))], code) # Simple fallback
        return parts

    def _get_distance(self, parts1, parts2):
        if parts1[0] != parts2[0]: return 1 # Section mismatch
        if parts1[1] != parts2[1]: return 2 # Class mismatch
        if parts1[2] != parts2[2]: return 3 # Subclass mismatch
        return 0 # Exact match

    def forward(self, logits, labels):
        base_loss = self.bce_loss(logits, labels.float()) # Ensure labels are float

        batch_size = logits.size(0)
        total_penalty = 0.0
        non_empty_samples = 0

        with torch.no_grad():
            # Get the index of the highest logit for each sample in the batch
            preds_indices = torch.argmax(logits, dim=1)

        for i in range(batch_size):
            # Find the indices where the true label is 1 (or > 0.5)
            true_indices = torch.where(labels[i] > 0.5)[0]

            # If there are no true labels for this sample, skip penalty calculation
            if len(true_indices) == 0:
                continue

            non_empty_samples += 1
            pred_idx = preds_indices[i].item() # The single predicted class index

            # Only apply penalty if the top prediction is NOT among the true labels
            if pred_idx not in true_indices:
                pred_parts = self.class_parts.get(pred_idx)
                # If predicted index somehow doesn't map to parts (shouldn't happen if classes list is correct), skip
                if pred_parts is None: continue

                min_distance = 4 # Max possible distance + 1 initialize
                # Find the minimum hierarchical distance between the predicted class
                # and ALL of the true classes for this sample.
                for true_idx_tensor in true_indices:
                    true_idx = true_idx_tensor.item()
                    true_parts = self.class_parts.get(true_idx)
                    # If true index somehow doesn't map to parts, skip comparison
                    if true_parts is None: continue
                    distance = self._get_distance(pred_parts, true_parts)
                    min_distance = min(min_distance, distance)

                # Add the penalty corresponding to the minimum distance found
                # Use .get with default 0.0 in case min_distance remained 4 (error case)
                total_penalty += self.distance_weights.get(min_distance, 0.0)

        # Average the penalty over samples that had at least one true label
        if non_empty_samples > 0:
             avg_penalty = total_penalty / non_empty_samples
        else:
             avg_penalty = 0.0 # No penalty if no samples had true labels

        # Ensure penalty is a tensor on the same device as base_loss
        hier_penalty = torch.tensor(avg_penalty, device=base_loss.device, dtype=base_loss.dtype)

        # Combine base loss and hierarchical penalty
        return base_loss + self.lambda_hier * hier_penalty


class HierarchicalLossTrainer(Trainer):
    def __init__(self, *args, hierarchical_loss_func=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the instantiated hierarchical loss function
        self.hierarchical_loss_func = hierarchical_loss_func

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.hierarchical_loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss