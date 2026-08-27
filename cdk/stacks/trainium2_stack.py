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
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from constructs import Construct

# Same AWS-published parameter as the trn1 stack; resolves per-region at deploy
# time. In sa-east-1 this was ami-0b1b0d3aaa2171e1f on 2026-08-03.
DLAMI_SSM_PARAM = "/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id"

# The artifacts bucket lives in us-west-2 and is NOT created here.
BUCKET_NAME = "neuron-pipelines-artifacts-600627330911"

USER_DATA_DIR = Path(__file__).resolve().parent.parent / "user_data"


def _strip_comments(text: str) -> str:
    """Drop whole-line comments and blank lines, keeping every shebang.

    EC2 caps user data at 16384 bytes RAW, and hunt mode concatenates four
    scripts. Adding the boot watchdog took the payload to 16887 and the deploy
    failed with InvalidUserData.Malformed.

    These files are deliberately comment-heavy -- the reasoning is the point,
    and several comments encode outages that cost real money. That reasoning
    belongs in the repo, where people read it; it does not need to be uploaded
    to a machine that only executes it. Stripping at synth time keeps both.

    Only WHOLE-LINE comments go. Trailing comments are left alone because `#`
    is legal inside strings, and this runs against shell that must still parse.
    Shebangs survive: heredocs here write executable scripts whose first line
    is load-bearing.
    """
    kept = [
        line
        for line in text.splitlines()
        if line.startswith("#!") or not line.lstrip().startswith("#")
        if line.strip()
    ]
    return "\n".join(kept)


def load_user_data(*scripts: str) -> ec2.UserData:
    """Concatenate user_data/*.sh files (read at synth time) into one script."""
    parts = [_strip_comments((USER_DATA_DIR / name).read_text()) for name in scripts]
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
        subnet_id = self.node.try_get_context("trn2SubnetId")
        az = self.node.try_get_context("trn2Az")

        # ---- Capacity Block (the only route that actually yielded hardware) ----
        # On-demand trn2.3xlarge capacity was empty in every AZ for ~10h, so the
        # box is launched into a purchased EC2 Capacity Block instead. That needs
        # TWO things a normal launch does not, and BOTH live on the launch
        # template because AWS::EC2::Instance cannot express market options:
        #
        #   InstanceMarketOptions.MarketType = capacity-block
        #   CapacityReservationTarget.CapacityReservationId = cr-...
        #
        # The block's InstanceMatchCriteria is "targeted": an untargeted launch
        # does NOT fall into it, it just fails on capacity like every other
        # attempt -- while the block bills for its whole window regardless.
        capacity_reservation_id = self.node.try_get_context("trn2CapacityReservationId")

        # ---- Scheduled launch (removes the laptop from the critical path) ----
        # A block opens at a fixed instant whether or not anyone is awake, and a
        # laptop-side launcher depends on the lid being open and on temporary
        # `aws login` credentials that expire. Both failed us once already, and
        # a missed window costs the whole upfront fee.
        #
        # So when trn2ScheduleLaunchAt is set the stack builds an Auto Scaling
        # group at desired=0 plus two ONE-TIME scheduled actions. AWS itself
        # scales it to 1 when the block opens. This is the pattern AWS documents
        # for Capacity Blocks, and its advantage over a one-shot RunInstances is
        # that Auto Scaling retries transient launch failures on its own.
        #
        # The scale-IN action is not optional bookkeeping. Per the Auto Scaling
        # guidance: "Scale in your Auto Scaling group to zero more than 30
        # minutes before the Capacity Block reservation end time. Amazon EC2 will
        # terminate any instances that are still running 30 minutes before the
        # end time." If we leave it running, EC2 reclaims it, the ASG reads that
        # as a failed health check, and tries to replace an instance it can never
        # get back.
        schedule_launch_at = self.node.try_get_context("trn2ScheduleLaunchAt")
        schedule_stop_at = self.node.try_get_context("trn2ScheduleStopAt")

        if schedule_launch_at and not capacity_reservation_id:
            raise ValueError(
                "trn2ScheduleLaunchAt is only meaningful with "
                "trn2CapacityReservationId -- there is nothing to schedule a "
                "launch into otherwise"
            )
        if schedule_launch_at and not schedule_stop_at:
            raise ValueError(
                "trn2ScheduleLaunchAt requires trn2ScheduleStopAt: an ASG left "
                "at desired=1 when the block ends fights EC2's reclaim with "
                "replacement launches that can never succeed"
            )

        if capacity_reservation_id and not (subnet_id and az):
            # A block lives in exactly one AZ. Defaulting the subnet here would
            # let a 1a deploy silently miss a 1b block and burn the window.
            raise ValueError(
                "trn2CapacityReservationId requires explicit trn2Az and "
                "trn2SubnetId matching the block's availability zone "
                "(cr-08dc8b22d254cd3da is in sa-east-1b / "
                "subnet-092833ea4d9c0210c)"
            )

        # ---- Capacity hunt: on-demand, unattended, and NOT a reservation -----
        # Quota is not capacity. On 2026-08-12, with 12 of 12 vCPU free, all
        # three sa-east-1 AZs returned InsufficientInstanceCapacity for an
        # on-demand trn2.3xlarge -- the same wall that forced the Capacity Block
        # a week earlier (108 watcher rounds, ~1.7h, nothing).
        #
        # A one-shot AWS::EC2::Instance is the wrong instrument against that: it
        # fails the stack and rolls back over minutes, so polling it is slow and
        # leaves wreckage. An Auto Scaling group at desired=1 retries launch
        # failures on its own, indefinitely, INSIDE AWS -- no laptop, no Lambda,
        # no poller to babysit -- and spanning every AZ that offers the type lets
        # it take whichever one opens first.
        #
        # Deliberately NOT an on-demand capacity reservation: those bill at the
        # full hourly rate from creation whether or not anything occupies them.
        # This mode holds nothing and costs nothing until a box actually starts.
        #
        # DO NOT "STOP ON CAPTURE". Stopping releases the physical hardware, and
        # the restart re-enters exactly the queue that took hours to clear -- so
        # a stopped trn2 is a lost trn2.
        #
        # trn2HuntStopAt IS OPTIONAL, AND OMITTING IT IS THE NORMAL MODE.
        # The first hunt ran with a 12h ceiling and made 95 launch attempts
        # across all three AZs, every one InsufficientInstanceCapacity, before
        # the ceiling disarmed it (2026-08-12/13). A ceiling expires on a date
        # chosen in advance, which has no relationship to when capacity actually
        # appears; the only thing it reliably does is switch the hunt off while
        # you are not looking. Failed launches are free, so an open-ended hunt
        # costs nothing to keep armed.
        #
        # What the ceiling used to bound was SPEND ONCE A BOX EXISTS, and that
        # bound has been moved onto the box, where it belongs: the gapfill
        # driver terminates from an EXIT trap, and cdk/user_data/trn2_hunt.sh
        # installs a 6h boot watchdog that terminates even if the driver never
        # starts. Set trn2HuntStopAt only when you want the hunt itself to give
        # up at a known instant.
        capacity_hunt = str(
            self.node.try_get_context("trn2CapacityHunt") or ""
        ).lower() in ("1", "true", "yes")
        hunt_stop_at = self.node.try_get_context("trn2HuntStopAt")

        if capacity_hunt and capacity_reservation_id:
            raise ValueError(
                "trn2CapacityHunt is an on-demand hunt and cannot target a "
                "Capacity Block. Drop trn2CapacityReservationId, or drop the hunt."
            )
        if capacity_hunt and schedule_launch_at:
            raise ValueError(
                "trn2CapacityHunt launches as soon as capacity exists; a "
                "scheduled launch time contradicts it"
            )

        subnet_id = subnet_id or "subnet-0489739583976c545"
        az = az or "sa-east-1a"

        # Every sa-east-1 AZ offers trn2.3xlarge. The hunt spans all three so the
        # group takes the first one to free up; the block path stays pinned to
        # the single AZ its reservation lives in.
        HUNT_SUBNETS = {
            "sa-east-1a": "subnet-0489739583976c545",
            "sa-east-1b": "subnet-092833ea4d9c0210c",
            "sa-east-1c": "subnet-088dd390f7aab1c53",
        }

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
        hunt_subnets = [
            ec2.Subnet.from_subnet_attributes(
                self, f"HuntSubnet{zone[-1].upper()}", subnet_id=sid, availability_zone=zone
            )
            for zone, sid in sorted(HUNT_SUBNETS.items())
        ]

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
                        # Let a hunt box END ITSELF when its lanes are done.
                        #
                        # Without this the box runs for maybe an hour and then
                        # bills until the scheduled ceiling hours later, which
                        # for an unattended capture is most of the cost. Plain
                        # `shutdown` is not enough either: the group would read
                        # the loss and launch a replacement, restarting the hunt.
                        # TerminateInstanceInAutoScalingGroup with
                        # ShouldDecrementDesiredCapacity does both halves in one
                        # call.
                        #
                        # Scoped by NAME PREFIX rather than the group's ARN on
                        # purpose: the ASG depends on this role, so referencing
                        # it here would be a circular dependency. The wildcard
                        # covers only CloudFormation's generated suffix.
                        iam.PolicyStatement(
                            sid="TerminateSelfWhenLanesFinish",
                            actions=[
                                "autoscaling:TerminateInstanceInAutoScalingGroup"
                            ],
                            resources=[
                                f"arn:aws:autoscaling:{self.region}:{self.account}"
                                ":autoScalingGroup:*:autoScalingGroupName/"
                                f"{construct_id}-*"
                            ],
                        ),
                    ]
                )
            },
        )

        machine_image = ec2.MachineImage.from_ssm_parameter(
            DLAMI_SSM_PARAM,
            os=ec2.OperatingSystemType.LINUX,
        )

        # In scheduled mode the box must also START the suite itself: moving the
        # launch into AWS but leaving the kickoff on a laptop would put the
        # laptop straight back on the critical path, and a Capacity Block pays
        # for idle minutes exactly like busy ones.
        # trn2_hunt.sh must come BEFORE trn2_autorun.sh: it writes the driver
        # selector that np-autorun reads. Both ASG modes auto-start, for the
        # same reason -- a box nobody is watching must not boot into an idle
        # bill -- but they start different drivers. See trn2_hunt.sh.
        user_data = load_user_data(
            "common.sh",
            "trn2.sh",
            *(("trn2_hunt.sh",) if capacity_hunt else ()),
            *(("trn2_autorun.sh",) if (schedule_launch_at or capacity_hunt) else ()),
        )

        block_devices = [
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
        ]

        # IMDSv2 + root volume via launch template: AWS::EC2::Instance's Ebs
        # mapping has no Throughput property, so gp3 throughput is only
        # expressible this way (same reason as the trn1 stack).
        #
        # In any ASG mode -- scheduled block OR on-demand capacity hunt -- the
        # template must ALSO carry the image, instance type, security group, role
        # and user data, because an Auto Scaling group launches from the template
        # alone with no ec2.Instance to supply them. Gating this on
        # schedule_launch_at alone left the hunt with a bare template and CDK
        # failed synth with "Setting 'launchTemplate' requires its 'instanceType'
        # to be set" -- the right error, for the right reason.
        launch_template = ec2.LaunchTemplate(
            self,
            "Trn2LaunchTemplate",
            require_imdsv2=True,
            block_devices=block_devices,
            **(
                dict(
                    machine_image=machine_image,
                    instance_type=ec2.InstanceType(instance_type),
                    security_group=security_group,
                    role=role,
                    user_data=user_data,
                )
                if (schedule_launch_at or capacity_hunt)
                else {}
            ),
        )

        if capacity_reservation_id:
            cfn_lt = launch_template.node.default_child
            cfn_lt.launch_template_data = ec2.CfnLaunchTemplate.LaunchTemplateDataProperty(
                **{
                    **cfn_lt.launch_template_data._values,
                    "instance_market_options": (
                        ec2.CfnLaunchTemplate.InstanceMarketOptionsProperty(
                            market_type="capacity-block",
                        )
                    ),
                    "capacity_reservation_specification": (
                        ec2.CfnLaunchTemplate.CapacityReservationSpecificationProperty(
                            capacity_reservation_target=(
                                ec2.CfnLaunchTemplate.CapacityReservationTargetProperty(
                                    capacity_reservation_id=capacity_reservation_id,
                                )
                            ),
                        )
                    ),
                }
            )

        Tags.of(self).add("Project", "neuron-pipelines")
        Tags.of(self).add("Stack", "trainium2")
        Tags.of(self).add("Lane", "training")

        if schedule_launch_at or capacity_hunt:
            # Mixed-instances groups and warm pools are NOT supported with
            # Capacity Blocks, so this stays a plain single-template ASG.
            asg = autoscaling.AutoScalingGroup(
                self,
                "Trn2Asg",
                vpc=vpc,
                # The hunt spreads across every AZ that offers the type; a block
                # is pinned to the one AZ its reservation physically lives in.
                vpc_subnets=ec2.SubnetSelection(
                    subnets=hunt_subnets if capacity_hunt else [subnet]
                ),
                launch_template=launch_template,
                min_capacity=0,
                max_capacity=1,
                # The hunt asks immediately and keeps asking until EC2 says yes.
                # The block path waits for its scheduled action to open the door.
                desired_capacity=1 if capacity_hunt else 0,
                # CDK defaults this to True, which emits
                # UpdatePolicy.AutoScalingScheduledAction.IgnoreUnmodifiedGroup-
                # SizeProperties and tells CloudFormation to skip pushing
                # min/max/desired whenever the TEMPLATE values are unchanged --
                # regardless of what the live group has drifted to. That is
                # correct for the Capacity Block path, where scheduled actions
                # own the sizes and a deploy must not fight them.
                #
                # It is actively wrong for the hunt. The live group is at
                # 0/0/0 because the old hunt-stop action scaled it in, while the
                # template still says 0/1/1 from the previous deploy -- so
                # "unchanged template" means CloudFormation pushes nothing, the
                # deploy reports success, and the hunt never actually arms.
                # This same interaction produced UPDATE_ROLLBACK_FAILED on
                # 2026-08-12. With no scheduled actions left in hunt mode there
                # is nothing to conflict with, so re-asserting the sizes on
                # every deploy is both safe and the only way `cdk deploy` means
                # "the hunt is armed".
                ignore_unmodified_size_properties=not capacity_hunt,
            )

            if schedule_launch_at:
                # One-time actions: CfnScheduledAction with a StartTime and no
                # Recurrence fires exactly once. min/max are pinned on both so a
                # scheduled action can never widen the group beyond the single
                # instance the block actually reserves.
                autoscaling.CfnScheduledAction(
                    self,
                    "Trn2ScheduleLaunch",
                    auto_scaling_group_name=asg.auto_scaling_group_name,
                    start_time=schedule_launch_at,
                    min_size=0,
                    max_size=1,
                    desired_capacity=1,
                )
                autoscaling.CfnScheduledAction(
                    self,
                    "Trn2ScheduleStop",
                    auto_scaling_group_name=asg.auto_scaling_group_name,
                    start_time=schedule_stop_at,
                    min_size=0,
                    max_size=0,
                    desired_capacity=0,
                )

            if hunt_stop_at:
                # The spend ceiling. An unattended hunt can succeed at 03:00 and
                # bill until someone notices; this terminates the group at a
                # fixed instant whether or not anyone is watching. Results must
                # therefore be pushed to S3 as lanes finish, never held on the
                # box -- the same discipline the Capacity Block window enforced.
                autoscaling.CfnScheduledAction(
                    self,
                    "Trn2HuntStop",
                    auto_scaling_group_name=asg.auto_scaling_group_name,
                    start_time=hunt_stop_at,
                    min_size=0,
                    max_size=0,
                    desired_capacity=0,
                )
                CfnOutput(self, "HuntStopAt", value=hunt_stop_at)

            # Instances inherit this at launch; phase2_status.sh and the
            # kickoff both find the box by Name tag.
            Tags.of(asg).add("Name", "neuron-pipelines-trn2")

            CfnOutput(self, "AutoScalingGroupName", value=asg.auto_scaling_group_name)
            CfnOutput(self, "LaunchMode", value="capacity-hunt (on-demand)"
                      if capacity_hunt else "scheduled capacity block")
            if schedule_launch_at:
                CfnOutput(self, "ScheduledLaunchAt", value=schedule_launch_at)
                CfnOutput(self, "ScheduledStopAt", value=schedule_stop_at)
            CfnOutput(
                self,
                "FindInstance",
                value=(
                    f"aws ec2 describe-instances --region {self.region} "
                    "--filters Name=tag:Name,Values=neuron-pipelines-trn2 "
                    "Name=instance-state-name,Values=pending,running "
                    "--query 'Reservations[0].Instances[0].InstanceId' --output text"
                ),
                description="The ASG launches the box; look it up by tag",
            )
            return

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
            user_data=user_data,
        )
        cfn_instance = instance.node.default_child
        cfn_instance.launch_template = (
            ec2.CfnInstance.LaunchTemplateSpecificationProperty(
                launch_template_id=launch_template.launch_template_id,
                version=launch_template.latest_version_number,
            )
        )

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
