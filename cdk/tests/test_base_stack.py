"""NeuronPipelinesBase template assertions (no AWS calls)."""
from aws_cdk.assertions import Match


def test_bucket_encrypted_blocked_versioned(base_template):
    base_template.has_resource_properties(
        "AWS::S3::Bucket",
        {
            "BucketName": "neuron-pipelines-artifacts-600627330911",
            "BucketEncryption": {
                "ServerSideEncryptionConfiguration": [
                    {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            },
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            },
            "VersioningConfiguration": {"Status": "Enabled"},
        },
    )


def test_bucket_lifecycle_and_retained(base_template):
    base_template.has_resource(
        "AWS::S3::Bucket",
        {
            "DeletionPolicy": "Retain",
            "Properties": Match.object_like(
                {
                    "LifecycleConfiguration": {
                        "Rules": Match.array_with(
                            [
                                Match.object_like(
                                    {
                                        "AbortIncompleteMultipartUpload": {
                                            "DaysAfterInitiation": 7
                                        },
                                        "Status": "Enabled",
                                    }
                                ),
                                Match.object_like(
                                    {
                                        "NoncurrentVersionExpiration": {
                                            "NoncurrentDays": 30
                                        },
                                        "Status": "Enabled",
                                    }
                                ),
                            ]
                        )
                    }
                }
            ),
        },
    )


def test_bucket_enforces_ssl(base_template):
    base_template.has_resource_properties(
        "AWS::S3::BucketPolicy",
        {
            "PolicyDocument": Match.object_like(
                {
                    "Statement": Match.array_with(
                        [
                            Match.object_like(
                                {
                                    "Effect": "Deny",
                                    "Action": "s3:*",
                                    "Condition": {
                                        "Bool": {"aws:SecureTransport": "false"}
                                    },
                                }
                            )
                        ]
                    )
                }
            )
        },
    )


def test_budget_amount_and_notifications(base_template):
    base_template.resource_count_is("AWS::Budgets::Budget", 1)
    base_template.has_resource_properties(
        "AWS::Budgets::Budget",
        {
            "Budget": Match.object_like(
                {
                    "BudgetType": "COST",
                    "TimeUnit": "MONTHLY",
                    "BudgetLimit": {"Amount": 200, "Unit": "USD"},
                }
            ),
            "NotificationsWithSubscribers": [
                {
                    "Notification": {
                        "NotificationType": "ACTUAL",
                        "ComparisonOperator": "GREATER_THAN",
                        "Threshold": 50,
                        "ThresholdType": "PERCENTAGE",
                    },
                    "Subscribers": [
                        {
                            "SubscriptionType": "EMAIL",
                            "Address": "alphacoma18@gmail.com",
                        }
                    ],
                },
                Match.object_like(
                    {"Notification": Match.object_like({"Threshold": 80})}
                ),
                Match.object_like(
                    {
                        "Notification": Match.object_like(
                            {"NotificationType": "ACTUAL", "Threshold": 100}
                        )
                    }
                ),
                Match.object_like(
                    {
                        "Notification": Match.object_like(
                            {"NotificationType": "FORECASTED", "Threshold": 100}
                        )
                    }
                ),
            ],
        },
    )


def test_security_group_has_zero_ingress_rules(base_template):
    sgs = base_template.find_resources("AWS::EC2::SecurityGroup")
    assert len(sgs) == 1
    for sg in sgs.values():
        props = sg["Properties"]
        assert not props.get("SecurityGroupIngress")
    # and no standalone ingress-rule resources either
    assert not base_template.find_resources("AWS::EC2::SecurityGroupIngress")


def test_no_wildcard_resources_in_inline_policies(base_template):
    statements = []
    # role.inline_policies land on the Role resource itself ...
    for role in base_template.find_resources("AWS::IAM::Role").values():
        for policy in role["Properties"].get("Policies", []):
            statements += policy["PolicyDocument"]["Statement"]
    # ... but guard against grant()-style default policies too
    for policy in base_template.find_resources("AWS::IAM::Policy").values():
        statements += policy["Properties"]["PolicyDocument"]["Statement"]

    assert statements, "expected at least one inline policy statement"
    for stmt in statements:
        resources = stmt.get("Resource", [])
        if not isinstance(resources, list):
            resources = [resources]
        assert "*" not in resources, f"wildcard Resource in statement: {stmt}"
