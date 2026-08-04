"""NeuronPipelinesTrainium2 template assertions (no AWS calls).

Mirrors test_trainium_stack.py, plus the assertions that only matter because
this box is in a DIFFERENT region from the artifacts bucket and the Base stack.
"""
from aws_cdk.assertions import Match

DLAMI_SSM_PARAM = "/aws/service/neuron/dlami/pytorch-2.9/ubuntu-24.04/latest/image_id"
BUCKET_NAME = "neuron-pipelines-artifacts-600627330911"


def test_instance_type_is_the_only_small_trainium2(trainium2_template):
    trainium2_template.resource_count_is("AWS::EC2::Instance", 1)
    trainium2_template.has_resource_properties(
        "AWS::EC2::Instance", {"InstanceType": "trn2.3xlarge"}
    )


def test_imdsv2_required(trainium2_template):
    trainium2_template.has_resource_properties(
        "AWS::EC2::LaunchTemplate",
        {
            "LaunchTemplateData": Match.object_like(
                {"MetadataOptions": Match.object_like({"HttpTokens": "required"})}
            )
        },
    )


def test_no_ssh_key(trainium2_template):
    for instance in trainium2_template.find_resources("AWS::EC2::Instance").values():
        assert "KeyName" not in instance["Properties"]
    for lt in trainium2_template.find_resources("AWS::EC2::LaunchTemplate").values():
        assert "KeyName" not in lt["Properties"]["LaunchTemplateData"]


def test_root_volume_500_gp3_3000_250_encrypted(trainium2_template):
    trainium2_template.has_resource_properties(
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
    lts = trainium2_template.find_resources("AWS::EC2::LaunchTemplate")
    (lt_logical_id,) = lts
    trainium2_template.has_resource_properties(
        "AWS::EC2::Instance",
        {"LaunchTemplate": Match.object_like({"LaunchTemplateId": {"Ref": lt_logical_id}})},
    )


def test_ami_resolves_from_the_same_neuron_ssm_parameter_as_trn1(trainium2_template):
    """No AMI pin: the SSM param resolves per-region at deploy time.

    A hardcoded id would be wrong here by construction -- AMI ids are regional,
    and the trn1 lane's ids do not exist in sa-east-1.
    """
    parameters = trainium2_template.to_json().get("Parameters", {})
    ssm_params = {
        logical_id: param
        for logical_id, param in parameters.items()
        if param.get("Default") == DLAMI_SSM_PARAM
        and param.get("Type", "").startswith("AWS::SSM::Parameter::Value")
    }
    assert ssm_params, f"no CFN parameter references {DLAMI_SSM_PARAM}"

    (logical_id,) = ssm_params
    trainium2_template.has_resource_properties(
        "AWS::EC2::Instance", {"ImageId": {"Ref": logical_id}}
    )


def test_tags(trainium2_template):
    for key, value in (
        ("Project", "neuron-pipelines"),
        ("Stack", "trainium2"),
        ("Lane", "training"),
    ):
        trainium2_template.has_resource_properties(
            "AWS::EC2::Instance",
            {"Tags": Match.array_with([{"Key": key, "Value": value}])},
        )


def test_user_data_includes_common_and_trn2_scripts(trainium2_template):
    instances = trainium2_template.find_resources("AWS::EC2::Instance")
    (instance,) = instances.values()
    user_data = instance["Properties"]["UserData"]["Fn::Base64"]
    assert "/opt/np/.userdata-done" in user_data       # common.sh
    assert "np-scratch.service" in user_data           # trn2.sh
    assert "NP_DEVICE=trn2" in user_data               # profile selection
    assert "NP_CACHE_PREFIX=neuron-cache-v3" in user_data


def test_user_data_has_no_swapfile(trainium2_template):
    """128 GiB of host RAM: the trn1 64 GiB swapfile would be dead weight.

    Pinned as a test because copying trn1.sh wholesale is the obvious mistake.
    """
    instances = trainium2_template.find_resources("AWS::EC2::Instance")
    (instance,) = instances.values()
    user_data = instance["Properties"]["UserData"]["Fn::Base64"]
    assert "swapfile" not in user_data
    assert "mkswap" not in user_data


def test_creates_no_bucket_and_reaches_the_us_west_2_one_by_arn(trainium2_template):
    """Results, code and the NEFF cache stay in ONE bucket across three boxes.

    Creating a second bucket here would split make_report.py's inputs; the
    us-west-2 bucket is therefore referenced by ARN, which is legal because IAM
    is global.
    """
    trainium2_template.resource_count_is("AWS::S3::Bucket", 0)
    policies = trainium2_template.find_resources("AWS::IAM::Role")
    (role,) = policies.values()
    doc = role["Properties"]["Policies"][0]["PolicyDocument"]
    resources = str(doc)
    assert BUCKET_NAME in resources


def test_iam_has_no_wildcard_resource(trainium2_template):
    """Same posture as BaseStack: never a bare Resource:"*"."""
    (role,) = trainium2_template.find_resources("AWS::IAM::Role").values()
    for statement in role["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]:
        assert statement.get("Resource") != "*"
        if statement["Sid"] == "DecryptViaSsmOnly":
            # KMS key wildcard is allowed ONLY under the ViaService condition.
            assert "kms:ViaService" in str(statement["Condition"])


def test_security_group_is_egress_only(trainium2_template):
    """SSM Session Manager needs zero inbound rules."""
    (sg,) = trainium2_template.find_resources("AWS::EC2::SecurityGroup").values()
    assert not sg["Properties"].get("SecurityGroupIngress")


# ---------------------------------------------------------------------------
# Capacity Blocks
#
# On-demand trn2.3xlarge capacity was empty in every sa-east-1 AZ for ~10h, so
# the box launches into a purchased EC2 Capacity Block. That needs BOTH a
# capacity-block market type and a TARGETED reservation id -- the block's
# InstanceMatchCriteria is "targeted", so a launch that names only one of the
# two does not fall into the block, it fails on capacity like every other
# attempt while the block bills for its entire window regardless.
# ---------------------------------------------------------------------------

CR_ID = "cr-08dc8b22d254cd3da"
CR_CONTEXT = {
    "trn2CapacityReservationId": CR_ID,
    "trn2Az": "sa-east-1b",
    "trn2SubnetId": "subnet-092833ea4d9c0210c",
}


def _capacity_block_template():
    from aws_cdk.assertions import Template

    from conftest import make_stacks

    return Template.from_stack(make_stacks(CR_CONTEXT)[3])


def _launch_template_data(template):
    (lt,) = template.find_resources("AWS::EC2::LaunchTemplate").values()
    return lt["Properties"]["LaunchTemplateData"]


def test_no_capacity_block_wiring_by_default(trainium2_template):
    """Without the context key this stays an ordinary on-demand launch.

    A stray capacity-block market type on a normal deploy would fail the launch
    outright, so the default path must carry neither field.
    """
    data = _launch_template_data(trainium2_template)
    assert "InstanceMarketOptions" not in data
    assert "CapacityReservationSpecification" not in data


def test_capacity_block_sets_market_type_and_targets_the_reservation():
    data = _launch_template_data(_capacity_block_template())
    assert data["InstanceMarketOptions"] == {"MarketType": "capacity-block"}
    assert (
        data["CapacityReservationSpecification"]["CapacityReservationTarget"][
            "CapacityReservationId"
        ]
        == CR_ID
    )


def test_capacity_block_preserves_imdsv2_and_gp3_throughput():
    """The market/reservation fields are grafted onto LaunchTemplateData with an
    escape hatch, which rebuilds the whole property -- so assert the two things
    that only exist BECAUSE of the launch template are still there."""
    data = _launch_template_data(_capacity_block_template())
    assert data["MetadataOptions"]["HttpTokens"] == "required"
    (mapping,) = data["BlockDeviceMappings"]
    assert mapping["Ebs"]["VolumeType"] == "gp3"
    assert mapping["Ebs"]["Throughput"] == 250
    assert mapping["Ebs"]["Encrypted"] is True


def test_capacity_block_lands_in_the_blocks_availability_zone():
    template = _capacity_block_template()
    (instance,) = template.find_resources("AWS::EC2::Instance").values()
    assert instance["Properties"]["AvailabilityZone"] == "sa-east-1b"
    assert instance["Properties"]["SubnetId"] == "subnet-092833ea4d9c0210c"


def test_capacity_block_without_an_explicit_az_is_refused():
    """A block lives in exactly ONE az. Silently defaulting to sa-east-1a would
    miss a sa-east-1b block and burn the paid-for window on a failed launch."""
    import pytest

    from conftest import make_stacks

    with pytest.raises(ValueError, match="requires explicit trn2Az"):
        make_stacks({"trn2CapacityReservationId": CR_ID, "trn2Az": None,
                     "trn2SubnetId": None})
