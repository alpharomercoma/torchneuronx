"""Trainium2 stack: one trn2.3xlarge training box in sa-east-1 (São Paulo).

WHY THIS IS A SEPARATE STACK AND NOT A CONTEXT FLAG ON TrainiumStack
--------------------------------------------------------------------
It lives in a different REGION. TrainiumStack takes `base: BaseStack` and reads
`base.vpc` / `base.security_group` / `base.role` — CloudFormation cannot
reference resources across regions, so this stack must mint its own VPC lookup,
security group and instance role. The one thing it does NOT duplicate is the
artifacts bucket: results, code and the NEFF cache stay in the single us-west-2
bucket so `make pull-results` and analysis/make_report.py keep one source of
truth for all three boxes. The bucket is therefore referenced by ARN, not
created, and the S3 traffic is cross-region by design (a couple of GiB).

WHY sa-east-1
-------------
trn2.3xlarge — one Trainium2, the only small Trainium2 SKU — is offered ONLY in
sa-east-1 (verified 2026-08-03 via describe-instance-type-offerings across all
17 enabled regions). Quota L-2C3B7624 was granted at 12 vCPU there, which is
exactly one instance.

WHY NO AMI PIN
--------------
The same SSM parameter TrainiumStack uses resolves in-region to the sa-east-1
Neuron PyTorch 2.9 / Ubuntu 24.04 DLAMI. The load-bearing `ami-035c945d…` pin
in the runbook is an INFERENTIA concern (vLLM 0.16 vs the Trn2-only 0.21
kernels) and has nothing to do with this training box.

Everything else — IMDSv2 and gp3 throughput via an explicit LaunchTemplate,
SSM-only access with no key pair, user-data read at synth time — mirrors
TrainiumStack deliberately, so a difference between the two boxes is always a
deliberate one.
"""
from pathlib import Path

from aws_cdk import (
    CfnOutput,
    Stack,
    Tags,
)
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from constructs import Construct

# Same AWS-published parameter as the trn1 stack; resolves per-region at deploy
# time. In sa-east-1 this was ami-0b1b0d3aaa2171e1f on 2026-08-03.
DLAMI_SSM_PARAM = "/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id"

# The artifacts bucket lives in us-west-2 and is NOT created here.
BUCKET_NAME = "neuron-pipelines-artifacts-600627330911"

USER_DATA_DIR = Path(__file__).resolve().parent.parent / "user_data"


def load_user_data(*scripts: str) -> ec2.UserData:
    """Concatenate user_data/*.sh files (read at synth time) into one script."""
    parts = [(USER_DATA_DIR / name).read_text() for name in scripts]
    return ec2.UserData.custom("\n".join(parts))


class Trainium2Stack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- Tunable context values (override with `cdk deploy -c key=value`) ----
        instance_type = self.node.try_get_context("trn2InstanceType") or "trn2.3xlarge"
        volume_gb = int(self.node.try_get_context("trn2VolumeGb") or 500)
        # All three sa-east-1 AZs offer trn2.3xlarge, but quota is not capacity:
        # on InsufficientInstanceCapacity, redeploy with -c trn2Az=sa-east-1b
        # -c trn2SubnetId=subnet-092833ea4d9c0210c (see runbook 12).
        subnet_id = (
            self.node.try_get_context("trn2SubnetId") or "subnet-0489739583976c545"
        )
        az = self.node.try_get_context("trn2Az") or "sa-east-1a"

        # ---- Networking: the account's default sa-east-1 VPC (no NAT cost) ----
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        security_group = ec2.SecurityGroup(
            self,
            "Trn2InstanceSg",
            vpc=vpc,
            description="neuron-pipelines trn2: egress only (SSM Session Manager)",
            allow_all_outbound=True,
        )

        subnet = ec2.Subnet.from_subnet_attributes(
            self, "PinnedSubnet", subnet_id=subnet_id, availability_zone=az
        )

        # ---- IAM: same scoped shape as BaseStack, but cross-region on S3 ----
        # IAM is global, so the us-west-2 bucket ARN is addressable from here.
        # The SSM/KMS statements stay in THIS region: the HF token was
        # replicated into sa-east-1 as a SecureString so the box never reaches
        # across regions for a secret.
        role = iam.Role(
            self,
            "Trn2InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="neuron-pipelines trn2 box: SSM access + scoped S3/SSM",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore"
                ),
            ],
            inline_policies={
                "neuron-pipelines-scoped": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            sid="ArtifactObjects",
                            actions=[
                                "s3:GetObject",
                                "s3:PutObject",
                                "s3:DeleteObject",
                            ],
                            resources=[f"arn:aws:s3:::{BUCKET_NAME}/*"],
                        ),
                        iam.PolicyStatement(
                            sid="ArtifactBucketList",
                            actions=["s3:ListBucket"],
                            resources=[f"arn:aws:s3:::{BUCKET_NAME}"],
                        ),
                        # GetBucketLocation: the CLI needs it to sign requests
                        # against a bucket in a region other than its own.
                        iam.PolicyStatement(
                            sid="ArtifactBucketLocation",
                            actions=["s3:GetBucketLocation"],
                            resources=[f"arn:aws:s3:::{BUCKET_NAME}"],
                        ),
                        iam.PolicyStatement(
                            sid="PipelineParameters",
                            actions=["ssm:GetParameter"],
                            resources=[
                                f"arn:aws:ssm:{self.region}:{self.account}"
                                ":parameter/neuron-pipelines/*"
                            ],
                        ),
                        # SecureString parameters decrypt through SSM; scope the
                        # KMS grant to keys in this account/region AND to use via
                        # the SSM service only — never a bare Resource:"*".
                        iam.PolicyStatement(
                            sid="DecryptViaSsmOnly",
                            actions=["kms:Decrypt"],
                            resources=[
                                f"arn:aws:kms:{self.region}:{self.account}:key/*"
                            ],
                            conditions={
                                "StringEquals": {
                                    "kms:ViaService": (
                                        f"ssm.{self.region}.amazonaws.com"
                                    )
                                }
                            },
                        ),
                    ]
                )
            },
        )

        machine_image = ec2.MachineImage.from_ssm_parameter(
            DLAMI_SSM_PARAM,
            os=ec2.OperatingSystemType.LINUX,
        )

        # IMDSv2 + root volume via launch template: AWS::EC2::Instance's Ebs
        # mapping has no Throughput property, so gp3 throughput is only
        # expressible this way (same reason as the trn1 stack).
        launch_template = ec2.LaunchTemplate(
            self,
            "Trn2LaunchTemplate",
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
            "Trn2Instance",
            instance_type=ec2.InstanceType(instance_type),
            machine_image=machine_image,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnets=[subnet]),
            security_group=security_group,
            role=role,
            # no key pair: SSM Session Manager only
            user_data=load_user_data("common.sh", "trn2.sh"),
        )
        cfn_instance = instance.node.default_child
        cfn_instance.launch_template = (
            ec2.CfnInstance.LaunchTemplateSpecificationProperty(
                launch_template_id=launch_template.launch_template_id,
                version=launch_template.latest_version_number,
            )
        )

        Tags.of(self).add("Project", "neuron-pipelines")
        Tags.of(self).add("Stack", "trainium2")
        Tags.of(self).add("Lane", "training")
        Tags.of(instance).add("Name", "neuron-pipelines-trn2")

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
