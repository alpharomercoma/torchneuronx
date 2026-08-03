#!/usr/bin/env python3
"""CDK app entrypoint: neuron-pipelines infrastructure, two regions.

Four independently deployable stacks:

  us-west-2 (Oregon)
  * NeuronPipelinesBase       — S3 artifacts bucket, EC2 instance role/profile,
                                egress-only security group, monthly cost budget.
  * NeuronPipelinesTrainium   — one trn1 training box (Neuron DLAMI via SSM param).
  * NeuronPipelinesInferentia — one inf2 inference box (Neuron vLLM DLAMI lookup).

  sa-east-1 (São Paulo)
  * NeuronPipelinesTrainium2  — one trn2.3xlarge training box. Separate region
                                because that is the ONLY region on earth that
                                offers the small Trainium2 SKU, and separate
                                stack because CloudFormation cannot reference
                                the Base stack's VPC/SG/role across regions.
                                It still writes to the us-west-2 bucket, so all
                                three boxes report into one comparison.

The account/region are pinned (not taken from the environment) because both the
default-VPC lookup and the AMI lookup require a concrete env, and because the
artifacts bucket name embeds the account id.
"""
import aws_cdk as cdk

from stacks.base_stack import BaseStack
from stacks.inferentia_stack import InferentiaStack
from stacks.trainium2_stack import Trainium2Stack
from stacks.trainium_stack import TrainiumStack

ACCOUNT = "600627330911"
REGION = "us-west-2"
TRN2_REGION = "sa-east-1"

app = cdk.App()
env = cdk.Environment(account=ACCOUNT, region=REGION)
trn2_env = cdk.Environment(account=ACCOUNT, region=TRN2_REGION)

base = BaseStack(
    app,
    "NeuronPipelinesBase",
    env=env,
    description="neuron-pipelines shared plumbing: artifacts bucket, instance role, SG, budget",
)

trainium = TrainiumStack(
    app,
    "NeuronPipelinesTrainium",
    base=base,
    env=env,
    description="neuron-pipelines training lane: one trn1 instance on the Neuron PyTorch DLAMI",
)
trainium.add_dependency(base)

inferentia = InferentiaStack(
    app,
    "NeuronPipelinesInferentia",
    base=base,
    env=env,
    description="neuron-pipelines inference lane: one inf2 instance on the Neuron vLLM DLAMI",
)
inferentia.add_dependency(base)

# No add_dependency on base: it is in another region, and a cross-region
# dependency is not expressible. The coupling that DOES exist -- the bucket --
# is by ARN, so BaseStack must already be deployed. Runbook 12 says so.
trainium2 = Trainium2Stack(
    app,
    "NeuronPipelinesTrainium2",
    env=trn2_env,
    description="neuron-pipelines Trainium2 lane: one trn2.3xlarge in sa-east-1",
)

app.synth()
