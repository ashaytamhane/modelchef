"""One-instance instant EC2 Fleet allocation for a bounded Spot training run."""
import base64

from botocore.exceptions import ClientError


def launch(ec2, settings, plan, request):
    types = settings.get('fleet_instance_types', [])
    subnets = settings.get('fleet_subnet_ids', [])
    if (settings.get('market_type', 'spot') != 'spot' or not types or not subnets
            or any(t not in {'g5.xlarge', 'g6.xlarge', 'g6e.xlarge'} for t in types)):
        raise ValueError('Instant GPU Fleet requires Spot and explicit four-vCPU G5/G6/G6e pools')
    specifications = ec2.describe_instance_types(InstanceTypes=types)['InstanceTypes']
    if len(specifications) != len(set(types)) or any(
            t['VCpuInfo']['DefaultVCpus'] != 4 for t in specifications):
        raise ValueError('Every Fleet instance type must have exactly four vCPUs')
    networks = ec2.describe_subnets(SubnetIds=subnets)['Subnets']
    if len(networks) != len(set(subnets)) or any(
            s['VpcId'] != plan['vpc_id'] or s['State'] != 'available'
            or not s['AvailabilityZone'].startswith(settings['region']) for s in networks):
        raise ValueError('Fleet subnets must be available in the same authorized region and VPC')
    offerings = set()
    paginator = ec2.get_paginator('describe_instance_type_offerings')
    for page in paginator.paginate(LocationType='availability-zone', Filters=[
            {'Name': 'instance-type', 'Values': types}]):
        offerings.update((o['InstanceType'], o['Location']) for o in page['InstanceTypeOfferings'])
    overrides = [{'InstanceType': kind, 'SubnetId': subnet['SubnetId'],
                  'WeightedCapacity': 1, 'MaxPrice': str(settings['max_spot_hourly_usd'])}
                 for kind in types for subnet in networks
                 if (kind, subnet['AvailabilityZone']) in offerings]
    if not overrides:
        raise ValueError('No offered instance type matches the authorized Fleet pools')
    run_id = request['ClientToken']
    data = {key: request[key] for key in ['ImageId', 'MetadataOptions',
            'InstanceInitiatedShutdownBehavior', 'BlockDeviceMappings', 'TagSpecifications']}
    if 'IamInstanceProfile' in request:
        data['IamInstanceProfile'] = request['IamInstanceProfile']
    data['UserData'] = base64.b64encode(request['UserData'].encode()).decode()
    data['NetworkInterfaces'] = [{k: v for k, v in request['NetworkInterfaces'][0].items()
                                 if k != 'SubnetId'}]
    template = ec2.create_launch_template(ClientToken=run_id + '-template',
        LaunchTemplateName=run_id, LaunchTemplateData=data)['LaunchTemplate']['LaunchTemplateId']
    try:
        response = ec2.create_fleet(ClientToken=run_id, Type='instant',
            TargetCapacitySpecification={'TotalTargetCapacity': 1, 'SpotTargetCapacity': 1,
                'OnDemandTargetCapacity': 0, 'DefaultTargetCapacityType': 'spot'},
            SpotOptions={'AllocationStrategy': 'price-capacity-optimized',
                         'SingleInstanceType': False, 'SingleAvailabilityZone': False},
            LaunchTemplateConfigs=[{'LaunchTemplateSpecification': {
                'LaunchTemplateId': template, 'Version': '1'}, 'Overrides': overrides}])
        ids = [instance for pool in response.get('Instances', []) for instance in pool['InstanceIds']]
        if len(ids) == 1:
            return {'Instances': [{'InstanceId': ids[0]}]}
        if ids:
            raise RuntimeError('Fleet returned an unexpected instance count; reconcile tagged instances')
        errors = response.get('Errors', [])
        capacity_codes = {'InsufficientInstanceCapacity', 'capacity-not-available',
                          'InsufficientCapacity', 'UnfulfillableCapacity'}
        code = ('InsufficientInstanceCapacity' if errors and all(
            e['ErrorCode'] in capacity_codes for e in errors) else
            (errors[0]['ErrorCode'] if errors else 'UnconfirmedFleetResult'))
        raise ClientError({'Error': {'Code': code, 'Message': str(errors)}}, 'CreateFleet')
    finally:
        # An instant Fleet does not replenish capacity or need the template afterward.
        ec2.delete_launch_template(LaunchTemplateId=template)
