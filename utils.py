from typing import Any,Dict,List,Optional,Tuple,Union
import torch
import torch.nn.functional as F


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: Union[int, float], dim: int = -1) -> torch.Tensor:
    if tensor.size(dim) >= length:
        return tensor
    else:
        pad_size = list(tensor.shape)
        pad_size[dim] = length - tensor.size(dim)
        return torch.cat(
            [
                tensor,
                pad_value * torch.ones(*pad_size, dtype=tensor.dtype, device=tensor.device),
            ],
            dim=dim,
        )
    

def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    is_encoder_decoder: bool = False,
    label_pad_token_id: int = -100,
    padding_value: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        is_encoder_decoder: Whether the model is an encoder-decoder model.
        label_pad_token_id: The label pad token id.
        padding_value: The padding value to use for the concatenated inputs_ids.
        device: The device for the concatenated inputs.

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    concatenated_batch = {}

    if is_encoder_decoder:
        max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
    else:
        max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

    for k in batch:
        if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
            if "labels" in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith("_input_ids"):
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            concatenated_key = k.replace("chosen", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
            if "labels" in k or is_encoder_decoder:
                pad_value = label_pad_token_id
            elif k.endswith("_input_ids"):
                pad_value = padding_value
            elif k.endswith("_attention_mask"):
                pad_value = 0
            concatenated_key = k.replace("rejected", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            ).to(device=device)

    if is_encoder_decoder:
        concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
        concatenated_batch["concatenated_attention_mask"] = (
            batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
        )

    concatenated_batch["concatenated_labels"]=concatenated_batch["concatenated_input_ids"]
    return concatenated_batch


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta=0.1,
    device=None,
    loss_type="sigmoid",
    label_smoothing=0
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    pi_logratios = pi_logratios.to(device)
    ref_logratios = ref_logratios.to(device)
    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "hinge":
        losses = torch.relu(1 -beta * logits)
    elif loss_type == "ipo":
        # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "kto_pair":
        # eqn (7) of the HALOs paper
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
        losses = torch.cat(
            (
                1 - F.sigmoid(beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
        )

    chosen_rewards = (
        beta
        * (
            policy_chosen_logps.to(device) - reference_chosen_logps.to(device)
        ).detach()
    )
    rejected_rewards = (
        beta
        * (
            policy_rejected_logps.to(device)
            - reference_rejected_logps.to(device)
        ).detach()
    )

    return losses, chosen_rewards, rejected_rewards


def get_batch_logps(
      logits: torch.FloatTensor,
      labels: torch.LongTensor,
      average_log_prob: bool = False,
      label_pad_token_id: int = -100,
      is_encoder_decoder: bool = False,
  ) -> torch.FloatTensor:
      """Compute the log probabilities of the given labels under the given logits.

      Args:
          logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
          labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
          average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
          label_pad_token_id: The label pad token id.
          is_encoder_decoder: Whether the model is an encoder-decoder model.

      Returns:
          A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
      """
      if logits.shape[:-1] != labels.shape:
          raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

      if not is_encoder_decoder:
          labels = labels[:, 1:].clone()
          logits = logits[:, :-1, :]
      loss_mask = labels != label_pad_token_id

      # dummy token; we'll ignore the losses on these tokens later
      labels[labels == label_pad_token_id] = 0

      per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
      del labels,logits
      if average_log_prob:
          return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
      else:
          return (per_token_logps * loss_mask).sum(-1)

def get_batch_loss_metrics(
    policy_logps,
    policy_logits,
    len_chosen,
    loss
):
    """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
    metrics = {}

    policy_chosen_logps,policy_rejected_logps = policy_logps[:len_chosen],policy_logps[len_chosen:]
    policy_chosen_logits,policy_rejected_logits= policy_logits[:len_chosen],policy_logits[len_chosen:]


    losses, chosen_rewards, rejected_rewards = loss
    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    prefix = "train_" 
    metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean()
    metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean()
    metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean()
    metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean()
    metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean()
    metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean()
    metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean()
    metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean()

    return losses.mean(), metrics

