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
from stacks.trainium2_stack import Trainium2Stack
from stacks.trainium_stack import TrainiumStack

ACCOUNT = "600627330911"
REGION = "us-west-2"
ENV = cdk.Environment(account=ACCOUNT, region=REGION)
SUBNET_ID = "subnet-0004804fe13fb126e"

# The Trainium2 box lives in sa-east-1 -- the only region offering trn2.3xlarge
# -- so its default-VPC lookup needs its own pre-seeded context key.
TRN2_REGION = "sa-east-1"
TRN2_ENV = cdk.Environment(account=ACCOUNT, region=TRN2_REGION)
TRN2_SUBNET_ID = "subnet-0489739583976c545"
TRN2_AZ = "sa-east-1a"


def _vpc_context_key(region):
    return (f"vpc-provider:account={ACCOUNT}:filter.isDefault=true:"
            f"region={region}:returnAsymmetricSubnets=true")


def _vpc_context_value(subnet_id, az):
    return {
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
                        "subnetId": subnet_id,
                        "cidr": "172.31.16.0/20",
                        "availabilityZone": az,
                        "routeTableId": "rtb-0aaaaaaaaaaaaaaa0",
                    }
                ],
            }
        ],
    }


VPC_CONTEXT_KEY = _vpc_context_key(REGION)
VPC_CONTEXT_VALUE = _vpc_context_value(SUBNET_ID, "us-west-2b")
TRN2_VPC_CONTEXT_KEY = _vpc_context_key(TRN2_REGION)
TRN2_VPC_CONTEXT_VALUE = _vpc_context_value(TRN2_SUBNET_ID, TRN2_AZ)

BASE_CONTEXT = {
    # mirrors cdk.json
    "az": "us-west-2b",
    "subnetId": SUBNET_ID,
    "volumeGb": 500,
    "budgetUsd": 200,
    "alertEmail": "alphacoma18@gmail.com",
    "trn1InstanceType": "trn1.2xlarge",
    "inf2InstanceType": "inf2.xlarge",
    "trn2InstanceType": "trn2.3xlarge",
    "trn2SubnetId": TRN2_SUBNET_ID,
    "trn2Az": TRN2_AZ,
    "trn2VolumeGb": 500,
    "@aws-cdk/aws-ec2:launchTemplateDefaultUserData": True,
    "@aws-cdk/core:checkSecretUsage": True,
    "@aws-cdk/aws-iam:minimizePolicies": True,
    "@aws-cdk/core:validateSnapshotRemovalPolicy": True,
    # pre-seeded lookup results so tests never call AWS (one per region)
    VPC_CONTEXT_KEY: VPC_CONTEXT_VALUE,
    TRN2_VPC_CONTEXT_KEY: TRN2_VPC_CONTEXT_VALUE,
}


def make_stacks(extra_context: dict | None = None):
    """Fresh app + all four stacks, mirroring app.py wiring."""
    app = cdk.App(context={**BASE_CONTEXT, **(extra_context or {})})
    base = BaseStack(app, "NeuronPipelinesBase", env=ENV)
    trainium = TrainiumStack(app, "NeuronPipelinesTrainium", base=base, env=ENV)
    inferentia = InferentiaStack(app, "NeuronPipelinesInferentia", base=base, env=ENV)
    # No base= : the Trainium2 stack is in another region and mints its own
    # VPC lookup, security group and role. See stacks/trainium2_stack.py.
    trainium2 = Trainium2Stack(app, "NeuronPipelinesTrainium2", env=TRN2_ENV)
    return base, trainium, inferentia, trainium2


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


@pytest.fixture(scope="session")
def trainium2_template(default_stacks):
    return Template.from_stack(default_stacks[3])
