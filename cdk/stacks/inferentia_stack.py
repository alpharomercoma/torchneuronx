"""Inferentia stack: one inf2 inference box on the Neuron vLLM DLAMI.

Design decisions:
  * There is no AWS-published SSM parameter for the "PyTorch Inference vLLM"
    DLAMI flavor, so the AMI is found via ec2.MachineImage.lookup() (name
    pattern + owner amazon). The lookup result is CACHED in cdk.context.json —
    the AMI is pinned until you `cdk context --reset` it, which is what you
    want for reproducible inference boxes.
  * Escape hatch: `-c inf2AmiId=ami-...` bypasses the lookup entirely and uses
    that exact AMI (useful when the name pattern drifts or for pinning).
  * Instance type comes from context `inf2InstanceType` (default inf2.xlarge,
    which fits an 8-vCPU Inf quota; inf2.2xlarge / inf2.8xlarge are one-flag
    upgrades once quota allows).
  * Same posture as the trainium lane: IMDSv2 required, no key pair, SG + role
    from NeuronPipelinesBase, pinned subnet, 500 GiB encrypted gp3 root.
    IMDSv2 + root volume live in an explicit LaunchTemplate because
    AWS::EC2::Instance Ebs mappings cannot express gp3 Throughput.
  * User data = common.sh + inf2.sh (inf2 has no NVMe instance store, so
    inf2.sh is a placeholder).
"""
from aws_cdk import (
    CfnOutput,
    Stack,
    Tags,
)
from aws_cdk import aws_ec2 as ec2
from constructs import Construct

from stacks.base_stack import BaseStack
from stacks.trainium_stack import load_user_data

# Matches the AWS "Deep Learning AMI Neuron PyTorch Inference vLLM (Ubuntu 24.04)"
# series; lookup picks the most recent by creation date.
DLAMI_NAME_PATTERN = "Deep Learning AMI Neuron PyTorch Inference vLLM*Ubuntu 24.04*"


class InferentiaStack(Stack):
    def __init__(
        self, scope: Construct, construct_id: str, *, base: BaseStack, **kwargs
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- Tunable context values (override with `cdk deploy -c key=value`) ----
        instance_type = self.node.try_get_context("inf2InstanceType") or "inf2.xlarge"
        volume_gb = int(self.node.try_get_context("volumeGb") or 500)
        subnet_id = self.node.try_get_context("subnetId") or "subnet-0004804fe13fb126e"
        az = self.node.try_get_context("az") or "us-west-2b"
        ami_override = self.node.try_get_context("inf2AmiId")  # escape hatch

        subnet = ec2.Subnet.from_subnet_attributes(
            self, "PinnedSubnet", subnet_id=subnet_id, availability_zone=az
        )

        if ami_override:
            machine_image = ec2.MachineImage.generic_linux(
                {self.region: ami_override}
            )
        else:
            machine_image = ec2.MachineImage.lookup(
                name=DLAMI_NAME_PATTERN,
                owners=["amazon"],
            )

        # IMDSv2 + root volume via launch template (see module docstring).
        launch_template = ec2.LaunchTemplate(
            self,
            "Inf2LaunchTemplate",
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
            "Inf2Instance",
            instance_type=ec2.InstanceType(instance_type),
            machine_image=machine_image,
            vpc=base.vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
            security_group=base.security_group,
            role=base.role,
            # no key pair: SSM Session Manager only
            user_data=load_user_data("common.sh", "inf2.sh"),
        )
        cfn_instance = instance.node.default_child
        cfn_instance.launch_template = (
            ec2.CfnInstance.LaunchTemplateSpecificationProperty(
                launch_template_id=launch_template.launch_template_id,
                version=launch_template.latest_version_number,
            )
        )

        Tags.of(self).add("Project", "neuron-pipelines")
        Tags.of(self).add("Stack", "inferentia")
        Tags.of(self).add("Lane", "inference")
        Tags.of(instance).add("Name", "neuron-pipelines-inf2")

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
