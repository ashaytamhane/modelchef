# Starter ingredients

These paragraphs and question/answer pairs were authored for ModelChef. They are small software-test inputs, not a scraped corpus, benchmark or useful instruction-tuning dataset.

Run from the repository root:

```bash
modelchef pipeline configs/pretrain-demo.yaml
modelchef pipeline configs/pretrain-demo.yaml --run
# Or pretrain, instruction tune and export in one workflow:
modelchef cook configs/workflow-demo.yaml --run
```

After dependencies are installed, these runs use local files and make no dataset/model downloads or AWS calls. The recipe forces CPU FP32 for portability; choose another accelerator in your own recipe after checking `doctor`.
