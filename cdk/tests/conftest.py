"""Shared test plumbing: build the app with pre-seeded lookup context.

Vpc.from_lookup / MachineImage.lookup normally hit AWS at synth time and cache
into cdk.context.json. Tests must make no AWS calls, so the app is constructed
with the vpc-provider context key pre-seeded (the standard trick); the AMI
lookup falls back to the framework's dummy value ("ami-1234") when its context
is absent, which is fine for template assertions.
"""
import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from stacks.base_stack import BaseStack
from stacks.inferentia_stack import InferentiaStack
from stacks.trainium_stack import TrainiumStack

ACCOUNT = "600627330911"
REGION = "us-west-2"
ENV = cdk.Environment(account=ACCOUNT, region=REGION)
SUBNET_ID = "subnet-0004804fe13fb126e"

VPC_CONTEXT_KEY = (
    f"vpc-provider:account={ACCOUNT}:filter.isDefault=true:"
    f"region={REGION}:returnAsymmetricSubnets=true"
)
VPC_CONTEXT_VALUE = {
    "vpcId": "vpc-0aaaaaaaaaaaaaaa0",
    "vpcCidrBlock": "172.31.0.0/16",
    "ownerAccountId": ACCOUNT,
    "availabilityZones": [],
    "subnetGroups": [
        {
            "name": "Public",
            "type": "Public",
            "subnets": [
                {
                    "subnetId": SUBNET_ID,
                    "cidr": "172.31.16.0/20",
                    "availabilityZone": "us-west-2b",
                    "routeTableId": "rtb-0aaaaaaaaaaaaaaa0",
                }
            ],
        }
    ],
}

BASE_CONTEXT = {
    # mirrors cdk.json
    "az": "us-west-2b",
    "subnetId": SUBNET_ID,
    "volumeGb": 500,
    "budgetUsd": 200,
    "alertEmail": "alphacoma18@gmail.com",
    "trn1InstanceType": "trn1.2xlarge",
    "inf2InstanceType": "inf2.xlarge",
    "@aws-cdk/aws-ec2:launchTemplateDefaultUserData": True,
    "@aws-cdk/core:checkSecretUsage": True,
    "@aws-cdk/aws-iam:minimizePolicies": True,
    "@aws-cdk/core:validateSnapshotRemovalPolicy": True,
    # pre-seeded lookup result so tests never call AWS
    VPC_CONTEXT_KEY: VPC_CONTEXT_VALUE,
}


def make_stacks(extra_context: dict | None = None):
    """Fresh app + all three stacks, mirroring app.py wiring."""
    app = cdk.App(context={**BASE_CONTEXT, **(extra_context or {})})
    base = BaseStack(app, "NeuronPipelinesBase", env=ENV)
    trainium = TrainiumStack(app, "NeuronPipelinesTrainium", base=base, env=ENV)
    inferentia = InferentiaStack(app, "NeuronPipelinesInferentia", base=base, env=ENV)
    return base, trainium, inferentia


@pytest.fixture(scope="session")
def default_stacks():
    return make_stacks()


@pytest.fixture(scope="session")
def base_template(default_stacks):
    return Template.from_stack(default_stacks[0])


@pytest.fixture(scope="session")
def trainium_template(default_stacks):
    return Template.from_stack(default_stacks[1])


@pytest.fixture(scope="session")
def inferentia_template(default_stacks):
    return Template.from_stack(default_stacks[2])
