"""Base stack: everything the trn1/inf2 lanes share.

Design decisions:
  * Zero-secret access posture: instances get AmazonSSMManagedInstanceCore for
    Session Manager; there are no SSH keys and no ingress rules anywhere.
  * The artifacts bucket is RETAINed on stack delete — training checkpoints and
    compiled-model artifacts are the expensive thing here, the compute is cattle.
  * Inline IAM policy is scoped to exactly the bucket, the /neuron-pipelines/*
    SSM parameter namespace, and KMS decrypt *via the SSM service only* (for
    SecureString parameters). Deliberately NO Resource:"*" statements.
  * Egress-only security group on the account's default VPC (no NAT cost, no
    custom VPC to babysit).
  * A monthly CfnBudget with email alerts at 50/80/100% actual and 100%
    forecasted acts as the blast-radius guardrail for forgotten instances.
"""
from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import aws_budgets as budgets
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_s3 as s3
from constructs import Construct

BUCKET_NAME = "neuron-pipelines-artifacts-600627330911"


class BaseStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- Tunable context values (override with `cdk deploy -c key=value`) ----
        budget_usd = int(self.node.try_get_context("budgetUsd") or 200)
        alert_email = self.node.try_get_context("alertEmail") or "alphacoma18@gmail.com"

        # ---- Networking: reuse the account's default VPC (no NAT cost) ----
        self.vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        # Egress-only: Session Manager needs no inbound rules at all.
        self.security_group = ec2.SecurityGroup(
            self,
            "InstanceSg",
            vpc=self.vpc,
            description="neuron-pipelines instances: egress only (SSM Session Manager)",
            allow_all_outbound=True,
        )

        # ---- Artifacts bucket (checkpoints, compiled NEFFs, benchmark results) ----
        self.bucket = s3.Bucket(
            self,
            "ArtifactsBucket",
            bucket_name=BUCKET_NAME,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="abort-incomplete-multipart-uploads",
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                ),
                s3.LifecycleRule(
                    id="expire-noncurrent-versions",
                    noncurrent_version_expiration=Duration.days(30),
                ),
            ],
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ---- Instance role: SSM core + tightly scoped inline permissions ----
        self.role = iam.Role(
            self,
            "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="neuron-pipelines EC2 instances (trn1 + inf2 lanes)",
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
                            resources=[self.bucket.arn_for_objects("*")],
                        ),
                        iam.PolicyStatement(
                            sid="ArtifactBucketList",
                            actions=["s3:ListBucket"],
                            resources=[self.bucket.bucket_arn],
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

        # Shared instance profile (ec2.Instance also mints its own from the role;
        # this one exists for out-of-band launches, e.g. spot experiments).
        self.instance_profile = iam.InstanceProfile(
            self,
            "InstanceProfile",
            role=self.role,
        )

        # ---- Monthly cost budget: the "forgot to stop the box" alarm ----
        actual_thresholds = (50, 80, 100)
        budgets.CfnBudget(
            self,
            "MonthlyBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name="neuron-pipelines-monthly",
                budget_type="COST",
                time_unit="MONTHLY",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=budget_usd, unit="USD"
                ),
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="ACTUAL",
                        comparison_operator="GREATER_THAN",
                        threshold=pct,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL", address=alert_email
                        )
                    ],
                )
                for pct in actual_thresholds
            ]
            + [
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        notification_type="FORECASTED",
                        comparison_operator="GREATER_THAN",
                        threshold=100,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="EMAIL", address=alert_email
                        )
                    ],
                )
            ],
        )

        # ---- Outputs ----
        CfnOutput(self, "BucketName", value=self.bucket.bucket_name)
        CfnOutput(self, "InstanceRoleArn", value=self.role.role_arn)
        CfnOutput(self, "SecurityGroupId", value=self.security_group.security_group_id)
