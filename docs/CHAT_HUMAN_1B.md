# A human-conversation chatbot from scratch

Goal: short, relevant conversation about everyday life, hobbies and feelings.
This research recipe starts a fresh approximately 1B native decoder. It imports
no pretrained weights, uses no teacher model and includes no reasoning traces.
It is a small-data experiment, not a demonstrated general-purpose assistant.

## Ingredients

| Source | Purpose | Terms |
| --- | --- | --- |
| [PersonaChat](https://arxiv.org/abs/1801.07243), via [bavard's truecased mirror](https://huggingface.co/datasets/bavard/personachat_truecased) | Human-written multi-turn conversation and profile conditioning | CC-BY-4.0; preserve original authors and mirror transformations |
| [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues), via [brianist's mirror](https://huggingface.co/datasets/brianist/empathetic_dialogues) | Human conversation about emotional situations | CC-BY-NC-4.0; research/noncommercial use |

The curation recipe contains citations and authors. Source limits cover up to
140,000 PersonaChat rows and 80,000 EmpatheticDialogues rows, not that many
independent conversations. PersonaChat retains the gold response, conversation
history and supplied profile; distinct turns sharing a profile remain grouped
for splitting. The empathy adapter currently selects the first listener reply
per complete conversation; it does not retain every empathy turn. Neither
ingredient is a large general-language corpus. Retained counts and token sizes
must be measured before choosing the final training duration.

## Recipe

[workflow-chat-human-1b.yaml](../configs/workflow-chat-human-1b.yaml) curates data
and runs two stages:

1. **Conversation language:** next-token training from random initialization on
   full transcripts, with a custom byte-level BPE tokenizer.
2. **Assistant replies:** continue those same weights with response-masked
   supervised training. Questions and history provide context; only assistant
   reply targets contribute to this stage's loss. This is not fine-tuning an
   externally pretrained model.

The model retains the proven 40-layer, width-1,536, GQA/SwiGLU architecture,
8,192-token requested vocabulary and 2,048-token context. Training uses BF16,
activation checkpointing, microbatch one and the experimental 8-bit
optimizer-in-backward policy. Checkpoints are scheduled every 1,000 updates.
The stage recipes provisionally specify 50,000 foundation and 5,000 reply
updates, with 48-hour per-invocation cutoffs. These are bounds, not evidence
that this amount of training will produce a good chatbot.

The reply stage explicitly specifies its architecture/tokenizer settings to
avoid inheriting the foundation stage's document-packing mode. Parent weights
and tokenizer assets still come from the preceding stage.

## First step: curate and inspect, without training

Run in WSL from the repository:

```bash
python -m modelchef.cli curate configs/data-chat-human.yaml
python -m modelchef.cli curate configs/data-chat-human.yaml --run
```

The first command previews; the second acquires a bounded corpus with a 512 MB
download cap and creates `data/raw/chat-human-1b/attribution.json`,
`conversation-text.jsonl` and `conversation-pairs.jsonl`. Do not infer quality
from the cap: inspect actual retained counts, lengths and representative turns.
Source SHA256 manifests identify the acquired snapshots. No AWS is launched.

Then preview the whole recipe:

```bash
python -m modelchef.cli cook configs/workflow-chat-human-1b.yaml
```

Only after inspecting the data and a short GPU pilot, start long training:

```bash
export MODELCHEF_LOCAL_CHECKPOINT_KEEP=1
export MODELCHEF_DATA_ATTRIBUTION="$PWD/data/raw/chat-human-1b/attribution.json"
python -m modelchef.cli cook configs/workflow-chat-human-1b.yaml --run
```

For the completed reply-stage run directory:

```bash
python -m modelchef.cli chat RUN_DIR --backend vllm --max-tokens 128
```

`RUN_DIR` must be replaced by the actual directory printed by the workflow.
Use a supplied role-play profile to test PersonaChat-style behavior; also test
ordinary questions with no system prompt. The model may imitate human personal
experiences from the corpus, so it is not automatically an honest AI assistant.

## What counts as progress

Test held-out greetings, topic relevance, user-name recall within a conversation,
follow-up questions, emotional response appropriateness and repetition. Change
the user message and check that the response changes appropriately. Exclude
those evaluation conversations from both stages. Check exact prompt boundaries
and native/export equivalence before interpreting failures as training issues.

An initial cached-data audit retained 100 PersonaChat turns and 33 empathy
replies, with no `<think>` markers. A tiny CPU protocol run exercised tokenizer
training, checkpoint saving, resume and reload; it did not validate 1B chatbot
quality. Full corpus acquisition and a 1B conversation pilot remain next steps.
