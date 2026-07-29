"""NeuronPipelinesTrainium template assertions (no AWS calls)."""
from aws_cdk.assertions import Match

DLAMI_SSM_PARAM = "/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id"


def test_instance_type(trainium_template):
    trainium_template.resource_count_is("AWS::EC2::Instance", 1)
    trainium_template.has_resource_properties(
        "AWS::EC2::Instance", {"InstanceType": "trn1.2xlarge"}
    )


def test_imdsv2_required(trainium_template):
    trainium_template.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {
            "LaunchTemplateData": Match.object_like(
                {"MetadataOptions": Match.object_like({"HttpTokens": "required"})}
            )
        },
    )


def test_no_ssh_key(trainium_template):
    for instance in trainium_template.find_resources("AWS::EC2::Instance").values():
        assert "KeyName" not in instance["Properties"]
    for lt in trainium_template.find_resources("AWS::EC2::LaunchTemplate").values():
        assert "KeyName" not in lt["Properties"]["LaunchTemplateData"]


def test_root_volume_500_gp3_3000_250_encrypted(trainium_template):
    # gp3 Throughput is only expressible via a launch template — CFN's
    # AWS::EC2::Instance Ebs mapping has no Throughput property.
    trainium_template.has_resource_properties(
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
    # ... and the instance actually uses that launch template.
    lts = trainium_template.find_resources("AWS::EC2::LaunchTemplate")
    (lt_logical_id,) = lts
    trainium_template.has_resource_properties(
        "AWS::EC2::Instance",
        {"LaunchTemplate": Match.object_like({"LaunchTemplateId": {"Ref": lt_logical_id}})},
    )


def test_ami_resolves_from_neuron_ssm_parameter(trainium_template):
    """ImageId must be a Ref to a CFN parameter backed by the Neuron SSM param."""
    parameters = trainium_template.to_json().get("Parameters", {})
    ssm_params = {
        logical_id: param
        for logical_id, param in parameters.items()
        if param.get("Default") == DLAMI_SSM_PARAM
        and param.get("Type", "").startswith("AWS::SSM::Parameter::Value")
    }
    assert ssm_params, f"no CFN parameter references {DLAMI_SSM_PARAM}"

    (logical_id,) = ssm_params
    trainium_template.has_resource_properties(
        "AWS::EC2::Instance", {"ImageId": {"Ref": logical_id}}
    )


def test_tags(trainium_template):
    for key, value in (
        ("Project", "neuron-pipelines"),
        ("Stack", "trainium"),
        ("Lane", "training"),
    ):
        trainium_template.has_resource_properties(
            "AWS::EC2::Instance",
            {"Tags": Match.array_with([{"Key": key, "Value": value}])},
        )


def test_user_data_includes_common_and_trn1_scripts(trainium_template):
    instances = trainium_template.find_resources("AWS::EC2::Instance")
    (instance,) = instances.values()
    user_data = instance["Properties"]["UserData"]["Fn::Base64"]
    assert "/opt/np/.userdata-done" in user_data  # common.sh
    assert "np-scratch.service" in user_data  # trn1.sh
