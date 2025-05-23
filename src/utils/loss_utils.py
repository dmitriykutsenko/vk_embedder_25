import torch
import torch.nn.functional as F

def info_nce_loss(anchors: torch.Tensor,
                  positives: torch.Tensor,
                  batch_negs: torch.Tensor,
                  queue_embs: torch.Tensor,
                  temperature: float) -> torch.Tensor:
    
    anchors = F.normalize(anchors, dim=-1)
    positives = F.normalize(positives, dim=-1)

    refs = [positives]
    if batch_negs.numel():
        refs.append(F.normalize(batch_negs, dim=-1))
    if queue_embs.numel():
        refs.append(F.normalize(queue_embs, dim=-1))
    refs = torch.cat(refs, dim=0)  

    logits = torch.matmul(anchors, refs.T) / temperature 
    targets = torch.arange(anchors.size(0), device=anchors.device)

    loss = F.cross_entropy(logits, targets)
    return loss