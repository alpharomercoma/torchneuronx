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


# ---------------------------------------------------------------------------
# Scheduled launch (ASG mode)
#
# A Capacity Block opens at a fixed instant whether or not anyone is awake, and
# the laptop-side launcher depends on an open lid and on `aws login` credentials
# that expire. Both had already failed once. In scheduled mode AWS itself scales
# the group to 1 when the block opens, and the box starts the suite from
# user-data -- no operator in the loop at any point.
# ---------------------------------------------------------------------------

SCHEDULE_CONTEXT = {
    **CR_CONTEXT,
    "trn2ScheduleLaunchAt": "2026-08-05T11:31:00Z",
    "trn2ScheduleStopAt": "2026-08-06T10:50:00Z",
}


def _scheduled_template():
    from aws_cdk.assertions import Template

    from conftest import make_stacks

    return Template.from_stack(make_stacks(SCHEDULE_CONTEXT)[3])


def test_scheduled_mode_replaces_the_instance_with_an_asg():
    """An ec2.Instance cannot be scheduled; the ASG is the whole point."""
    template = _scheduled_template()
    template.resource_count_is("AWS::EC2::Instance", 0)
    template.resource_count_is("AWS::AutoScaling::AutoScalingGroup", 1)


def test_scheduled_asg_starts_empty_and_is_capped_at_one():
    """Capacity Blocks reserve an exact instance count. A group that could scale
    past it would launch instances the block cannot hold."""
    (asg,) = _scheduled_template().find_resources(
        "AWS::AutoScaling::AutoScalingGroup"
    ).values()
    assert asg["Properties"]["MinSize"] == "0"
    assert asg["Properties"]["MaxSize"] == "1"
    assert asg["Properties"]["DesiredCapacity"] == "0"


def test_scheduled_actions_are_one_time_and_bracket_the_window():
    """Two one-time actions: scale out at the block start, and scale IN before
    EC2 reclaims. AWS guidance: 'Scale in your Auto Scaling group to zero more
    than 30 minutes before the Capacity Block reservation end time.' Leaving it
    at 1 makes the ASG fight the reclaim with replacement launches."""
    actions = _scheduled_template().find_resources(
        "AWS::AutoScaling::ScheduledAction"
    ).values()
    by_time = {a["Properties"]["StartTime"]: a["Properties"] for a in actions}
    assert set(by_time) == {"2026-08-05T11:31:00Z", "2026-08-06T10:50:00Z"}

    up = by_time["2026-08-05T11:31:00Z"]
    assert up["DesiredCapacity"] == 1 and up["MaxSize"] == 1
    down = by_time["2026-08-06T10:50:00Z"]
    assert down["DesiredCapacity"] == 0 and down["MaxSize"] == 0

    # No Recurrence == fires exactly once. A cron here would relaunch daily
    # into a block that no longer exists.
    for props in by_time.values():
        assert "Recurrence" not in props


def test_scheduled_stop_precedes_the_reclaim_deadline():
    """The block ends 2026-08-06T11:30Z and EC2 starts terminating at 11:00Z."""
    import datetime

    (down,) = [
        a["Properties"]
        for a in _scheduled_template()
        .find_resources("AWS::AutoScaling::ScheduledAction")
        .values()
        if a["Properties"]["DesiredCapacity"] == 0
    ]
    stop = datetime.datetime.fromisoformat(down["StartTime"].replace("Z", "+00:00"))
    reclaim = datetime.datetime(2026, 8, 6, 11, 0, tzinfo=datetime.timezone.utc)
    assert stop < reclaim


def test_scheduled_launch_template_is_self_sufficient():
    """An ASG launches from the template ALONE -- there is no ec2.Instance left
    to supply the image, type, security group, profile or user data."""
    data = _launch_template_data(_scheduled_template())
    assert data["InstanceType"] == "trn2.3xlarge"
    assert "ImageId" in data
    assert "SecurityGroupIds" in data
    assert "IamInstanceProfile" in data
    assert "UserData" in data
    # and it still targets the block
    assert data["InstanceMarketOptions"] == {"MarketType": "capacity-block"}
    assert (
        data["CapacityReservationSpecification"]["CapacityReservationTarget"][
            "CapacityReservationId"
        ]
        == CR_ID
    )


def test_scheduled_mode_autoruns_the_suite():
    """Scheduling the launch but not the kickoff would put the laptop straight
    back on the critical path and pay for an idle box."""
    import base64

    data = _launch_template_data(_scheduled_template())
    user_data = str(data["UserData"])
    assert "np-autorun" in user_data
    assert "run_phase3_trn2.sh" in user_data
    assert "trn2_deadline_push.sh" in user_data


def test_unscheduled_mode_does_not_autorun():
    """A manual deploy still gives a quiet box you can drive by hand."""
    (instance,) = _capacity_block_template().find_resources(
        "AWS::EC2::Instance"
    ).values()
    assert "np-autorun" not in str(instance["Properties"]["UserData"])


def test_schedule_without_a_reservation_is_refused():
    import pytest

    from conftest import make_stacks

    with pytest.raises(ValueError, match="nothing to schedule a launch into"):
        make_stacks({"trn2ScheduleLaunchAt": "2026-08-05T11:31:00Z"})


def test_schedule_without_a_stop_time_is_refused():
    """An ASG left at desired=1 fights EC2's end-of-block reclaim."""
    import pytest

    from conftest import make_stacks

    with pytest.raises(ValueError, match="requires trn2ScheduleStopAt"):
        make_stacks({**CR_CONTEXT, "trn2ScheduleLaunchAt": "2026-08-05T11:31:00Z"})
