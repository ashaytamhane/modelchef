# One-instance Spot allocation with EC2 Fleet

Manual zone retries can miss capacity available in another pool. An instant EC2
Fleet lets AWS choose among explicitly authorized instance types and subnets in
one request. ModelChef uses `price-capacity-optimized`, considering both capacity
and price. AWS does not guarantee that a Fleet request will succeed.

Add these optional fields to your existing `aws_test` configuration:

```yaml
aws_test:
  # Retain all existing AMI, recipe, profile, artifact, budget and runtime fields.
  region: us-east-1
  market_type: spot
  instance_type: g6.xlarge
  fleet_instance_types: [g5.xlarge, g6.xlarge, g6e.xlarge]
  fleet_subnet_ids: [REPLACE_WITH_SUBNET_A, REPLACE_WITH_SUBNET_B]
```

This is a partial configuration, not a complete runnable recipe. The base
`instance_type` and `subnet_id` provide the launch preview; the Fleet may choose a
different authorized pool. Its selected type and zone are recorded as
`actual_instance_type` and `actual_availability_zone` in the run report.

Use a training recipe that fits **all** configured GPUs. G5/G6 have less GPU
memory than G6e; smaller microbatches and gradient accumulation can preserve the
effective batch size. Measure throughput rather than assuming identical speed.

The implementation validates four vCPUs for every candidate, validates subnets
in the same region/VPC and filters overrides against AWS instance offerings. It
uses target capacity one, zero On-Demand capacity and Fleet type `instant`, which
does not automatically replace an interrupted instance. Existing S3 uploads,
budget reservations, guest shutdown and external AWS deadline remain active.
The per-instance hourly ceiling applies to every override. An existing recipe
without Fleet fields continues to use `RunInstances`.

The IAM principal needs `ec2:CreateLaunchTemplate`, `ec2:DeleteLaunchTemplate`,
`ec2:CreateFleet`, instance/subnet/type-offering read permissions and the existing
instance launch, tagging and `iam:PassRole` permissions. Temporary launch
templates are deleted after the request. This feature does not grant IAM access.

Recovery identifies Fleet instances through the unique Project/RunId tags;
Fleet client tokens are not assumed to become instance client tokens. Ambiguous
responses must be reconciled before retrying. Only confirmed capacity rejections
with completed cleanup can safely trigger another launch attempt.

References: [instant Fleet](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/instant-fleet.html),
[allocation examples](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-fleet-examples.html).
