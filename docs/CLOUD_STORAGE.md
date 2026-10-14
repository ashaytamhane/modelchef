# Cloud storage lifecycle

Compute shutdown and persistent storage deletion are separate operations. Runpod
Pods and AWS training instances are cleaned up at completion or their deadline.
Their persistent storage remains until explicitly deleted. Retained storage keeps
billing after compute shutdown; the run's runtime allowance does not cover
indefinite retention.

Runpod results remain under the reported `/workspace/RUN_ID` path on the network
volume. AWS training artifacts remain in the configured S3 prefix; AWS training
and serving EBS root disks default to `DeleteOnTermination=false`. Run reports
include storage locations and volume IDs returned by the launch response. Fleet
launches may require discovering retained EBS IDs from their `RunId` tags.

Automatic local results downloads default to disabled. Explicitly set
`download_results: true` in the provider configuration to request downloads.
Runpod also supports an explicit `runpod-job RUN_DIR --action fetch` while its
Pod is live. No workflow automatically deletes network volumes or S3 objects.
AWS disk deletion at instance termination requires the explicit configuration
`delete_storage_on_termination: true`.

Inspect or preview deletion of a named Runpod volume:

```bash
python -m modelchef.cli runpod-volume VOLUME_ID --config RUNPOD_CONFIG.yaml
python -m modelchef.cli runpod-volume VOLUME_ID --config RUNPOD_CONFIG.yaml --action delete
```

Permanently delete it only when explicitly requested:

```bash
python -m modelchef.cli runpod-volume VOLUME_ID --config RUNPOD_CONFIG.yaml --action delete --run
python -m modelchef.cli aws-volume vol-0123456789abcdef0 --region us-east-1 --action delete --run
```

Omit `--run` to preview AWS volume deletion. An attached EBS volume is refused;
detach it explicitly first. Deletion is verified by a subsequent provider read.
These commands do not delete buckets, S3 prefixes or unrelated storage.
