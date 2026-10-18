"""Response-masked SFT, reference-based DPO and bounded GRPO-inspired RLVR."""
import html
import re
import unicodedata

import torch
import torch.nn.functional as F

from modelchef.corpus import PII, process


def format_prompt(text, prompt_format='user_assistant'):
    if prompt_format == 'chat_transcript':
        return text.rstrip() + '\nAssistant:'
    return f"User: {text}\nAssistant:"


def prepare_structured(records, c, root):
    """Filter whole records, preserve field boundaries and isolate prompts across splits."""
    fields = c.stage.data_fields()
    by_index, inputs, prompts, redactions = {}, [], set(), 0
    for row in records:
        row = dict(row)
        for field in fields:
            value = html.unescape(unicodedata.normalize(c.processing.unicode_normalization, row[field]))
            if c.processing.strip_html:
                value = re.sub(r'<(?!/?(?:think|answer)\b)[^>]+>', ' ', value)
            row[field] = value.strip()
        if c.processing.pii == 'redact':
            for field in fields:
                row[field], count = PII.subn('[REDACTED]', row[field])
                redactions += count
        if c.stage.kind == 'dpo' and row['chosen'] == row['rejected']:
            raise ValueError('DPO chosen/rejected responses must differ')
        if c.stage.kind == 'rlvr' and c.stage.reward == 'integer' and not re.fullmatch(
                r'[+-]?\d+', row['answer'].strip()):
            raise ValueError('Integer verifier requires signed integer answer strings')
        identity = ' '.join(row['prompt'].casefold().split())
        if identity in prompts:
            continue
        prompts.add(identity)
        row['text'] = ' '.join(row[field] for field in fields)
        by_index[row['source_index']] = row
        inputs.append(row)
    splits, info = process(inputs, c, root)
    for name, rows in splits.items():
        splits[name] = [{**row, **{field: by_index[row['source_index']][field] for field in fields},
                         **({'split_group': by_index[row['source_index']]['split_group']}
                            if 'split_group' in by_index[row['source_index']] else {})}
                        for row in rows]
        if c.stage.reasoning_format == 'think_answer' and c.stage.kind in {'sft', 'distill'}:
            for row in splits[name]:
                row['response'] = f"<think>{row['reasoning']}</think>\n<answer>{row['answer']}</answer>"
    info['prompt_duplicates'] = len(records) - len(inputs)
    info['counts']['pii_redactions'] = info['counts'].get('pii_redactions', 0) + redactions
    return splits, info


def response_pair(prompt, response, tokenizer, context, prompt_format='user_assistant', add_bos=False):
    prefix = tokenizer.encode(format_prompt(prompt, prompt_format), add_special_tokens=False).ids
    if add_bos and (not prefix or prefix[0] != tokenizer.token_to_id('<bos>')):
        prefix.insert(0, tokenizer.token_to_id('<bos>'))
    answer = tokenizer.encode(' ' + response, add_special_tokens=False).ids + [tokenizer.token_to_id('<eos>')]
    # Do not silently turn a partial response into a complete supervised example.
    if len(prefix) + len(answer) > context + 1:
        raise ValueError('Structured example exceeds context; increase context or shorten data')
    ids = prefix + answer
    x = ids[:-1]
    y = [-100] * (len(prefix) - 1) + answer
    padding = context - len(x)
    return (x + [tokenizer.token_to_id('<pad>')] * padding, y + [-100] * padding)


def pack_structured(rows, tokenizer, c, token_limit, progress=None):
    parts = [[], [], [], []] if c.stage.kind == 'dpo' else [[], []]
    used, skipped = 0, 0
    for row in rows:
        fields = ['chosen', 'rejected'] if c.stage.kind == 'dpo' else [
            'answer' if c.stage.kind == 'rlvr' else 'response']
        try:
            pairs = [response_pair(row['prompt'], row[field], tokenizer, c.model.context_length,
                                   c.stage.prompt_format, add_bos=c.tokenizer.add_bos) for field in fields]
        except ValueError as error:
            if c.stage.oversized_policy != 'skip' or not str(error).startswith('Structured example exceeds context'):
                raise
            skipped += 1
            continue
        # Count each token in the prompt/response sequence once, including EOS.
        # The final supervised target marks the real sequence boundary even when
        # an imported tokenizer uses the same token for padding and EOS.
        size = sum(max(index for index, token in enumerate(y) if token != -100) + 2
                   for _, y in pairs)
        if used + size > token_limit:
            break
        used += size
        for index, pair in enumerate(pairs):
            for offset, values in enumerate(pair):
                parts[index * 2 + offset].append(values)
    if not parts[0]:
        raise ValueError('No structured examples fit the split token budget')
    if progress:
        progress(f'Structured packing: retained {len(parts[0])} complete examples; skipped {skipped} oversized examples.')
    return tuple(torch.tensor(part, dtype=torch.long) for part in parts), used


def token_logps(model, x, y):
    logits = model(x).float()
    mask = y != -100
    logps = logits.log_softmax(-1).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return logps * mask, mask


def sequence_logps(model, x, y):
    return token_logps(model, x, y)[0].sum(-1)


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta):
    margin = beta * ((chosen - rejected) - (reference_chosen - reference_rejected))
    return -F.logsigmoid(margin).mean()


def verified_reward(completion, expected, kind, reasoning_format='plain'):
    actual, expected = completion.strip(), expected.strip()
    if reasoning_format == 'think_answer':
        answers = re.findall(r'<answer>(.*?)</answer>', actual, flags=re.DOTALL)
        if len(answers) != 1:
            return 0.0
        actual = answers[0].strip()
    if kind == 'exact_match':
        return float(actual == expected)
    if not re.fullmatch(r'[+-]?\d+', expected):
        raise ValueError('Integer verifier requires signed integer answer strings')
    return float(bool(re.fullmatch(r'[+-]?\d+', actual)) and int(actual) == int(expected))


def group_advantages(rewards):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-6)


def grpo_loss(policy, old, reference, advantages, mask, clip, kl_coefficient):
    ratio = (policy - old).clamp(-20, 20).exp()
    advantage = advantages[:, None]
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
    difference = (reference - policy).clamp(-20, 20)
    kl = difference.exp() - difference - 1
    return ((-surrogate + kl_coefficient * kl) * mask).sum(-1).div(mask.sum(-1)).mean()


def rlvr_loss(model, reference, x, y, tokenizer, c):
    losses, rewards_seen, zero_groups, generated_tokens = [], [], 0, 0
    model.eval()  # Sampling and rescoring must use the same policy, without dropout.
    for source, targets in zip(x, y):
        first = (targets != -100).nonzero()[0].item()
        prefix = source[:first + 1].tolist()
        expected = tokenizer.decode(targets[targets != -100].tolist()).strip()
        max_new = min(c.stage.rollout_tokens, c.model.context_length + 1 - len(prefix))
        trajectories, rewards = [], []
        with torch.no_grad():
            for _ in range(c.stage.group_size):
                ids = list(prefix)
                for _ in range(max_new):
                    logits = model(torch.tensor([ids], device=x.device))[:, -1].float()
                    # Sample the unmodified policy distribution, matching the objective logps.
                    token = torch.multinomial(logits.softmax(-1), 1).item()
                    ids.append(token)
                    if token == tokenizer.token_to_id('<eos>'):
                        break
                trajectories.append(ids)
                generated_tokens += len(ids) - len(prefix)
                text = tokenizer.decode(ids[len(prefix):])
                rewards.append(verified_reward(text, expected, c.stage.reward, c.stage.reasoning_format))
        padded_x, padded_y = [], []
        for ids in trajectories:
            padding = c.model.context_length - len(ids) + 1
            padded_x.append(ids[:-1] + [tokenizer.token_to_id('<pad>')] * padding)
            padded_y.append([-100] * (len(prefix) - 1) + ids[len(prefix):] + [-100] * padding)
        rx = torch.tensor(padded_x, device=x.device)
        ry = torch.tensor(padded_y, device=x.device)
        rewards = torch.tensor(rewards, device=x.device)
        advantages = group_advantages(rewards)
        with torch.no_grad():
            old, mask = token_logps(model, rx, ry)
            ref, _ = token_logps(reference, rx, ry)
        policy, _ = token_logps(model, rx, ry)
        losses.append(grpo_loss(policy, old, ref, advantages, mask,
                                c.stage.clip_ratio, c.stage.kl_coefficient))
        rewards_seen.extend(rewards.tolist())
        zero_groups += int(rewards.std(unbiased=False).item() == 0)
    model.train()
    return torch.stack(losses).mean(), {'reward_mean': sum(rewards_seen) / len(rewards_seen),
                                      'zero_advantage_groups': zero_groups,
                                      'rollout_tokens': generated_tokens}


def evaluate_stage(model, reference, blocks, tokenizer, c, device, amp):
    if c.stage.kind not in {'dpo', 'rlvr'}:
        return None
    model.eval()
    count = (len(blocks[0]) if c.evaluation.batches == 'all' else
             min(len(blocks[0]), c.evaluation.batches * (c.evaluation.batch_size or c.training.micro_batch_size)))
    parts = [part[:count].to(device) for part in blocks]
    with torch.no_grad(), amp():
        if c.stage.kind == 'dpo':
            cx, cy, rx, ry = parts
            chosen, rejected = sequence_logps(model, cx, cy), sequence_logps(model, rx, ry)
            ref_c, ref_r = sequence_logps(reference, cx, cy), sequence_logps(reference, rx, ry)
            margin = (chosen - rejected) - (ref_c - ref_r)
            return {'examples': count, 'preference_accuracy': (margin > 0).float().mean().item(),
                    'policy_likelihood_accuracy': (chosen > rejected).float().mean().item(),
                    'reference_adjusted_margin': margin.mean().item(),
                    'dpo_loss': dpo_loss(chosen, rejected, ref_c, ref_r, c.stage.dpo_beta).item()}
        x, y = parts
        rewards = []
        for source, targets in zip(x, y):
            first = (targets != -100).nonzero()[0].item()
            ids = source[:first + 1].tolist()
            prefix = len(ids)
            expected = tokenizer.decode(targets[targets != -100].tolist()).strip()
            for _ in range(min(c.stage.rollout_tokens, c.model.context_length + 1 - prefix)):
                token = model(torch.tensor([ids], device=device))[:, -1].argmax(-1).item()
                ids.append(token)
                if token == tokenizer.token_to_id('<eos>'):
                    break
            rewards.append(verified_reward(tokenizer.decode(ids[prefix:]), expected,
                                           c.stage.reward, c.stage.reasoning_format))
        return {'examples': count, 'greedy_verified_accuracy': sum(rewards) / len(rewards),
                'verifier': c.stage.reward}


def objective(model, parts, c, reference=None, tokenizer=None):
    if c.stage.uses_distillation:
        x, y = parts
        logits = model(x).float()
        with torch.no_grad():
            teacher = reference(x).float()
        temperature = c.stage.distillation_temperature
        kl = F.kl_div((logits / temperature).log_softmax(-1),
                      (teacher / temperature).softmax(-1), reduction='none').sum(-1)
        mask = y != -100
        soft = kl[mask].mean() * temperature ** 2
        hard = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        alpha = c.stage.distillation_alpha
        return alpha * soft + (1 - alpha) * hard, {'teacher_kl': soft.item()}
    if c.stage.kind == 'dpo':
        cx, cy, rx, ry = parts
        with torch.no_grad():
            reference_chosen = sequence_logps(reference, cx, cy)
            reference_rejected = sequence_logps(reference, rx, ry)
        # One policy forward keeps DDP's unused-parameter traversal correct for sparse experts.
        logits = model(torch.cat((cx, rx))).float()
        targets = torch.cat((cy, ry))
        mask = targets != -100
        logps = logits.log_softmax(-1).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        chosen, rejected = (logps * mask).sum(-1).split(len(cx))
        return dpo_loss(chosen, rejected,
                        reference_chosen, reference_rejected, c.stage.dpo_beta), {}
    if c.stage.kind == 'rlvr':
        return rlvr_loss(model, reference, *parts, tokenizer, c)
    x, y = parts
    return F.cross_entropy(model(x).float().flatten(0, 1), y.flatten()), {}
