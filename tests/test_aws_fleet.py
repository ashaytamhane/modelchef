from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from botocore.session import Session
from botocore.validate import validate_parameters

from modelchef.aws_fleet import launch


class EC2:
    def __init__(self, response=None):
        self.response = response if response is not None else {'Instances': [{'InstanceIds': ['i-one']}]}
        self.deleted = False

    def describe_instance_types(self, **kwargs):
        return {'InstanceTypes': [{'VCpuInfo': {'DefaultVCpus': 4}} for _ in kwargs['InstanceTypes']]}

    def describe_subnets(self, **kwargs):
        return {'Subnets': [{'SubnetId': 'subnet-one', 'VpcId': 'vpc-one', 'State': 'available',
                            'AvailabilityZone': 'us-east-1a'}]}

    def get_paginator(self, name):
        return SimpleNamespace(paginate=lambda **kwargs: [{'InstanceTypeOfferings': [
            {'InstanceType': t, 'Location': 'us-east-1a'} for t in kwargs['Filters'][0]['Values']]}])

    def create_launch_template(self, **kwargs):
        shape = Session().get_service_model('ec2').operation_model('CreateLaunchTemplate').input_shape
        validate_parameters(kwargs, shape)
        self.template = kwargs
        return {'LaunchTemplate': {'LaunchTemplateId': 'lt-one'}}

    def create_fleet(self, **kwargs):
        shape = Session().get_service_model('ec2').operation_model('CreateFleet').input_shape
        validate_parameters(kwargs, shape)
        self.fleet = kwargs
        return self.response

    def delete_launch_template(self, **kwargs):
        self.deleted = True


def inputs():
    settings = {'fleet_instance_types': ['g5.xlarge', 'g6.xlarge', 'g6e.xlarge'],
                'fleet_subnet_ids': ['subnet-one'], 'region': 'us-east-1',
                'max_spot_hourly_usd': '3'}
    plan = {'vpc_id': 'vpc-one'}
    request = {'ClientToken': 'run-one', 'ImageId': 'ami-one', 'UserData': '#!/bin/bash\n',
               'MetadataOptions': {'HttpTokens': 'required'},
               'InstanceInitiatedShutdownBehavior': 'terminate', 'BlockDeviceMappings': [],
               'TagSpecifications': [], 'NetworkInterfaces': [{'DeviceIndex': 0,
               'SubnetId': 'subnet-one', 'Groups': ['sg-one'], 'AssociatePublicIpAddress': True}]}
    return settings, plan, request


def test_fleet_targets_exactly_one_spot_and_preserves_shutdown():
    client = EC2()
    assert launch(client, *inputs())['Instances'][0]['InstanceId'] == 'i-one'
    assert client.fleet['Type'] == 'instant'
    assert client.fleet['TargetCapacitySpecification']['TotalTargetCapacity'] == 1
    assert client.fleet['TargetCapacitySpecification']['OnDemandTargetCapacity'] == 0
    assert client.fleet['SpotOptions']['AllocationStrategy'] == 'price-capacity-optimized'
    assert len(client.fleet['LaunchTemplateConfigs'][0]['Overrides']) == 3
    assert client.template['LaunchTemplateData']['InstanceInitiatedShutdownBehavior'] == 'terminate'
    assert client.deleted


def test_capacity_rejection_is_retryable_and_template_deleted():
    client = EC2({'Errors': [{'ErrorCode': 'InsufficientInstanceCapacity'}]})
    with pytest.raises(ClientError) as caught:
        launch(client, *inputs())
    assert caught.value.response['Error']['Code'] == 'InsufficientInstanceCapacity'
    assert client.deleted


def test_empty_unconfirmed_response_is_not_a_capacity_rejection():
    client = EC2({})
    with pytest.raises(ClientError) as caught:
        launch(client, *inputs())
    assert caught.value.response['Error']['Code'] == 'UnconfirmedFleetResult'


def test_larger_instance_is_rejected_before_launch():
    settings, plan, request = inputs()
    settings['fleet_instance_types'] = ['g6e.2xlarge']
    with pytest.raises(ValueError, match='four-vCPU'):
        launch(EC2(), settings, plan, request)
