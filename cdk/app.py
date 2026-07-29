#!/usr/bin/env python3
"""CDK app entrypoint: neuron-pipelines infrastructure in Oregon (us-west-2).

Three independently deployable stacks:

  * NeuronPipelinesBase       — S3 artifacts bucket, EC2 instance role/profile,
                                egress-only security group, monthly cost budget.
  * NeuronPipelinesTrainium   — one trn1 training box (Neuron DLAMI via SSM param).
  * NeuronPipelinesInferentia — one inf2 inference box (Neuron vLLM DLAMI lookup).

The account/region are pinned (not taken from the environment) because both the
default-VPC lookup and the AMI lookup require a concrete env, and because the
artifacts bucket name embeds the account id.
"""
import aws_cdk as cdk

from stacks.base_stack import BaseStack
from stacks.inferentia_stack import InferentiaStack
from stacks.trainium_stack import TrainiumStack

ACCOUNT = "600627330911"
REGION = "us-west-2"

app = cdk.App()
env = cdk.Environment(account=ACCOUNT, region=REGION)

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

app.synth()
