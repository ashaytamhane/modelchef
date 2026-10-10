# First substantial from-scratch experiment

Aim first for coherent **short story continuations**. General chat, factual question
answering and instruction following are separate goals that require broader training
and suitable post-training data. This recipe is a starting experiment, not a proven
quality result or a reproduction of a published model.

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) provides synthetic
simple-English stories. The [original paper](https://arxiv.org/abs/2305.07759) studies
coherent language generation with small models. Review its CDLA-Sharing-1.0 dataset
license before training or distributing derived artifacts. ModelChef selects a bounded
ordered slice, not the full corpus or a representative random sample.

## Recipes

| File | Purpose |
| --- | --- |
| [pretrain-tinystories.yaml](../configs/pretrain-tinystories.yaml) | New BPE and random native decoder; about 21M parameters, 512-token context |
| [workflow-tinystories.yaml](../configs/workflow-tinystories.yaml) | Runs that stage and exports its trained weights |
| [aws-tinystories.yaml](../configs/aws-tinystories.yaml) | G6 Spot resources with $8 per-run / $10 overall reservations and six-hour deadline |

The training recipe requests up to 100,000 stories within a 1 GB response cap, and packs
up to 64M tokens. All weights and the 8,192-token BPE vocabulary start from scratch.
It requests 12,000 optimizer updates, microbatch 8 and accumulation 4: up to
`12000 × 8 × 4 × 512 = 196,608,000` token positions processed. This repeats sampled
training blocks; it is not an epoch iterator or a guarantee that each story is visited.
The actual corpus can contain fewer tokens after filtering and splitting.

Near-deduplication is disabled to avoid the memory-mode pairwise algorithm's cost at
this scale. Exact duplicate removal remains enabled; similar stories can still cross
splits. Held-out loss therefore needs that qualification. No automatic HellaSwag score
is requested for this narrow story-writing experiment.

## Preview, then launch explicitly

### Optional Hugging Face authentication and rate limits

The viewer accepts [Bearer-token authentication](https://huggingface.co/docs/dataset-viewer/quick_start).
Use a read token through `hf auth login` in the same WSL environment, or an externally
managed `HF_TOKEN` environment variable. Install the optional client if needed:

```bash
uv pip install 'huggingface_hub>=0.34,<1'
hf auth login
```

Never put the token in YAML, Git or chat. ModelChef reads the environment/local Hub
credential cache and attaches it to viewer requests; it does not save token values in
recipes or manifests. Authentication does not guarantee faster viewer responses or
remove rate limits. See [Hub rate limits](https://huggingface.co/docs/hub/rate-limits);
Hub file-download limits should not be assumed to equal viewer endpoint limits.

Viewer pages are paced two seconds apart. HTTP 429 and selected transient server errors
retry the same page up to six times with backoff and `Retry-After` handling. Messages
explain the actual cooldown. A server cooldown over five minutes stops instead of
retrying early. If acquisition ultimately fails, rerun after the cooldown: incomplete
acquisition is not yet resumable, so it starts again; `--resume` is for training
checkpoints. Only a complete acquisition is cached for subsequent reuse.

Credentials on your laptop are not forwarded to AWS guests. Public TinyStories can be
acquired anonymously there with the same retry/pacing logic; private datasets require
separately managed guest credentials or a prepared local input bundle.

### Configure cloud execution

Update your clone and install controller dependencies and appropriate PyTorch first:

```bash
git pull --ff-only
uv pip install -e '.[dev,pipeline,aws]'
cp configs/aws-tinystories.yaml configs/aws-tinystories.local.yaml
```

Edit the local AWS file: set your actual CLI profile and region, private artifact
bucket, existing training instance profile and any required subnet/AMI settings.
Follow [AWS access setup](../README.md#prepare-aws-access); placeholders do not create
that infrastructure. G6/L4 supports this CUDA/BF16 recipe, but verify the AMI driver and
Spot offering through the preview. The hourly ceiling is a limit, not a quoted price.

```bash
python -m modelchef.cli cook configs/workflow-tinystories.yaml --target aws --aws-config configs/aws-tinystories.local.yaml
# Only after reviewing resources, rates and the reservation:
python -m modelchef.cli cook configs/workflow-tinystories.yaml --target aws --aws-config configs/aws-tinystories.local.yaml --run
```

For one stage without final workflow export, use `pipeline configs/pretrain-tinystories.yaml`
with the same target/config flags. On an already prepared compatible GPU host, omit the
AWS flags to train there directly. This batch is selected for a cloud GPU; reduce it
deliberately for smaller devices. Hardware detection does not calibrate batch size.

Data acquisition happens on the guest. Its time and dependency setup count toward the
six-hour cloud deadline; the training loop has its own five-hour cap. Throughput for
this exact recipe has not been measured on G6. If the job hits a deadline before finishing,
it is incomplete: inspect returned checkpoints and reports rather than treating it as
a successful trained model. The safeguards estimate costs and attempt shutdown; AWS
billing remains authoritative, including retained S3 storage. No launch is performed
just by adding these files to the repository.

## Decide whether it learned

### Larger general-text paths

The included [approximately 150M](../configs/pretrain-dense-150m.yaml) and
[approximately 350M](../configs/pretrain-dense-350m.yaml) dense recipes expose a larger
general-text path. Matching [150M](../configs/workflow-dense-150m.yaml) and
[350M workflows](../configs/workflow-dense-350m.yaml) add final export. These are
ModelChef scale adaptations of the dense architecture discussed in the
[Llama 3 report](https://arxiv.org/abs/2407.21783), not its original data or hyperparameters.

Supply a licensed, curated local `data/raw/corpus.jsonl` first, or copy a recipe and
set `data.local_paths` for shards. Disk processing uses mapped token blocks. Defaults
request up to 8B packed tokens and 200,000 updates of 32,768 token positions each
(about 6.55B positions). The three-day training runtime cap can end the job sooner.
The generic generation prompt probes completion; it is not instruction supervision.

Measure tokens/sec and memory with a shorter personal calibration recipe. Estimate
training hours as token positions divided by measured tokens/sec and 3,600; add data
preparation, compilation, evaluation and transfer time. Set the AWS deadline and
storage/result caps for that whole job. The $10 starter budget is not a sufficient
funding claim for these runs. These recipes never raise cloud limits automatically.
Use `cook --target aws` with your configured cloud file; the launch review shows current
Spot price, runtime and reservation and asks you to type `yes` before launching.

Managed AWS currently transfers local corpora inside a capped in-memory source bundle.
Large local inputs can exceed the default bundle limit or the controller's RAM; choose
bounded shards and an appropriate limit rather than assuming disk-mode preprocessing
also streams the upload. Corpus-scale S3 ingestion is not implemented by this launcher.

### Evaluate the completed run

Confirm `status: completed`, loss history, checkpoint recovery and final export in the
returned reports. Compare initial and final held-out loss, and test multiple **unseen
story beginnings**, for example with `chat RUN_DIR --prompt "Once upon a time, a little fox found"`.
Use the recovered completed stage directory instead of the literal `RUN_DIR`. Inspect
the result archive before extracting, and retain snapshots/configuration with checkpoints.

Look for grammatical sentences, consistent characters and an ending that follows the
setup. Falling training loss alone can indicate memorization. Tune the data/step budget
after inspecting results; no specific response quality is guaranteed. To later teach
instruction behavior, supply genuine prompt/response data and add an SFT stage. The
included tiny SFT demo is not enough to turn this story model into a general assistant.
