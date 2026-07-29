"""Trainium stack: one trn1 training box on the Neuron PyTorch DLAMI.

Design decisions:
  * AMI comes from the AWS-published SSM parameter for the PyTorch 2.9 /
    Ubuntu 24.04 Neuron DLAMI, resolved at deploy time — no hardcoded AMI id,
    drivers/runtime stay current on every fresh deploy.
  * The subnet is pinned (context `subnetId`, default us-west-2b) because trn1
    capacity is not present in every AZ of us-west-2.
  * SSM-only access: require_imdsv2, no key pair, security group and role come
    from NeuronPipelinesBase (egress-only, scoped IAM).
  * Root volume 500 GiB gp3 @ 3000 IOPS / 250 MiB/s — model weights + Neuron
    compile cache are disk-hungry; delete_on_termination because artifacts that
    matter get synced to the RETAINed S3 bucket.
  * IMDSv2 + the root volume live in an explicit LaunchTemplate attached to the
    instance: CloudFormation's AWS::EC2::Instance Ebs mapping has NO Throughput
    property (the L2 silently drops it), so gp3 throughput can only be
    expressed via a launch template.
  * User data = user_data/common.sh + user_data/trn1.sh read at synth time.
    NOTE: changing user data REPLACES the instance (and its compile cache).
"""
from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Stack,
    Tags,
)
from aws_cdk import aws_ec2 as ec2
from constructs import Construct

from stacks.base_stack import BaseStack

# AWS-published SSM parameter that always points at the latest Neuron
# PyTorch 2.9 DLAMI for Ubuntu 24.04. Discover alternatives with:
#   aws ssm get-parameters-by-path --region us-west-2 --path /aws/service/neuron --recursive
DLAMI_SSM_PARAM = "/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id"

USER_DATA_DIR = Path(__file__).resolve().parent.parent / "user_data"


def load_user_data(*scripts: str) -> ec2.UserData:
    """Concatenate user_data/*.sh files (read at synth time) into one script."""
    parts = [(USER_DATA_DIR / name).read_text() for name in scripts]
    return ec2.UserData.custom("\n".join(parts))


class TrainiumStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, base: BaseStack, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- Tunable context values (override with `cdk deploy -c key=value`) ----
        instance_type = self.node.try_get_context("trn1InstanceType") or "trn1.2xlarge"
        volume_gb = int(self.node.try_get_context("volumeGb") or 500)
        subnet_id = self.node.try_get_context("subnetId") or "subnet-0004804fe13fb126e"
        az = self.node.try_get_context("az") or "us-west-2b"

        # Pin the exact subnet (trn1 capacity is AZ-sensitive in us-west-2).
        subnet = ec2.Subnet.from_subnet_attributes(
            self, "PinnedSubnet", subnet_id=subnet_id, availability_zone=az
        )

        machine_image = ec2.MachineImage.from_ssm_parameter(
            DLAMI_SSM_PARAM,
            os=ec2.OperatingSystemType.LINUX,
        )

        # IMDSv2 + root volume via launch template (see module docstring).
        launch_template = ec2.LaunchTemplate(
            self,
            "Trn1LaunchTemplate",
            require_imdsv2=True,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/sda1",  # Ubuntu DLAMI root device
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_gb,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        iops=3000,
                        throughput=250,
                        encrypted=True,
                        delete_on_termination=True,
                    ),
                )
            ],
        )

        instance = ec2.Instance(
            self,
            "Trn1Instance",
            instance_type=ec2.InstanceType(instance_type),
            machine_image=machine_image,
            vpc=base.vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
            security_group=base.security_group,
            role=base.role,
            # no key pair: SSM Session Manager only
            user_data=load_user_data("common.sh", "trn1.sh"),
        )
        cfn_instance = instance.node.default_child
        cfn_instance.launch_template = (
            ec2.CfnInstance.LaunchTemplateSpecificationProperty(
                launch_template_id=launch_template.launch_template_id,
                version=launch_template.latest_version_number,
            )
        )

        Tags.of(self).add("Project", "neuron-pipelines")
        Tags.of(self).add("Stack", "trainium")
        Tags.of(self).add("Lane", "training")
        Tags.of(instance).add("Name", "neuron-pipelines-trn1")

        # ---- Outputs ----
        CfnOutput(self, "InstanceId", value=instance.instance_id)
        CfnOutput(self, "AvailabilityZone", value=instance.instance_availability_zone)
        CfnOutput(
            self,
            "SsmConnect",
            value=(
                f"aws ssm start-session --region {self.region} "
                f"--target {instance.instance_id}"
            ),
            description="Connect without SSH keys via Session Manager",
        )
