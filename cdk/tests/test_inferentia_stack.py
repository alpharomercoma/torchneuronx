"""NeuronPipelinesInferentia template assertions (no AWS calls)."""
from aws_cdk.assertions import Match, Template

from conftest import make_stacks


def test_default_instance_type_inf2_xlarge(inferentia_template):
    inferentia_template.resource_count_is("AWS::EC2::Instance", 1)
    inferentia_template.has_resource_properties(
        "AWS::EC2::Instance", {"InstanceType": "inf2.xlarge"}
    )


def test_instance_type_context_override():
    """inf2InstanceType is context-switchable (8xlarge once quota lands)."""
    _, _, inferentia, _ = make_stacks({"inf2InstanceType": "inf2.8xlarge"})
    template = Template.from_stack(inferentia)
    template.has_resource_properties(
        "AWS::EC2::Instance", {"InstanceType": "inf2.8xlarge"}
    )


def test_imdsv2_required_and_no_ssh_key(inferentia_template):
    inferentia_template.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {
            "LaunchTemplateData": Match.object_like(
                {"MetadataOptions": Match.object_like({"HttpTokens": "required"})}
            )
        },
    )
    for instance in inferentia_template.find_resources("AWS::EC2::Instance").values():
        assert "KeyName" not in instance["Properties"]
    for lt in inferentia_template.find_resources("AWS::EC2::LaunchTemplate").values():
        assert "KeyName" not in lt["Properties"]["LaunchTemplateData"]


def test_root_volume_500_gp3_3000_250_encrypted(inferentia_template):
    # gp3 Throughput is only expressible via a launch template — CFN's
    # AWS::EC2::Instance Ebs mapping has no Throughput property.
    inferentia_template.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {
            "LaunchTemplateData": Match.object_like(
                {
                    "BlockDeviceMappings": [
                        Match.object_like(
                            {
                                "DeviceName": "/dev/sda1",
                                "Ebs": Match.object_like(
                                    {
                                        "VolumeSize": 500,
                                        "VolumeType": "gp3",
                                        "Iops": 3000,
                                        "Throughput": 250,
                                        "Encrypted": True,
                                        "DeleteOnTermination": True,
                                    }
                                ),
                            }
                        )
                    ]
                }
            )
        },
    )


def test_tags(inferentia_template):
    for key, value in (
        ("Project", "neuron-pipelines"),
        ("Stack", "inferentia"),
        ("Lane", "inference"),
    ):
        inferentia_template.has_resource_properties(
            "AWS::EC2::Instance",
            {"Tags": Match.array_with([{"Key": key, "Value": value}])},
        )


def test_inf2_ami_id_escape_hatch():
    """When -c inf2AmiId=... is set, the template uses exactly that AMI."""
    _, _, inferentia, _ = make_stacks({"inf2AmiId": "ami-0123456789abcdef0"})
    template = Template.from_stack(inferentia)
    template.has_resource_properties(
        "AWS::EC2::Instance", {"ImageId": "ami-0123456789abcdef0"}
    )


def test_user_data_includes_common_and_inf2_scripts(inferentia_template):
    instances = inferentia_template.find_resources("AWS::EC2::Instance")
    (instance,) = instances.values()
    user_data = instance["Properties"]["UserData"]["Fn::Base64"]
    assert "/opt/np/.userdata-done" in user_data  # common.sh
    assert "/opt/np/.userdata-inf2-done" in user_data  # inf2.sh
