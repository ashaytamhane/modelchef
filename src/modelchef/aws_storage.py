"""Explicit-only management of retained AWS training volumes."""
import re
import time


def manage_volume(volume_id, region, action='status', execute=False, profile=None, ec2=None):
    if not re.fullmatch(r'vol-[0-9a-f]{8,17}', volume_id):
        raise ValueError('Name a valid EBS volume ID')
    if action not in {'status', 'delete'}:
        raise ValueError('Volume action must be status or delete')
    if ec2 is None:
        import boto3
        ec2 = boto3.Session(profile_name=profile, region_name=region).client('ec2')
    volume = ec2.describe_volumes(VolumeIds=[volume_id])['Volumes'][0]
    if volume['VolumeId'] != volume_id:
        raise ValueError('Volume read-back identity differs')
    result = {'volume': volume, 'action': action, 'deleted': False,
              'note': 'Retained storage bills until explicit deletion; delete --run permanently removes data.'}
    if action == 'delete' and execute:
        if volume.get('Attachments') or volume['State'] != 'available':
            raise ValueError('Volume is attached or unavailable; detach it explicitly first')
        ec2.delete_volume(VolumeId=volume_id)
        from botocore.exceptions import ClientError
        for _ in range(10):
            try:
                ec2.describe_volumes(VolumeIds=[volume_id])
            except ClientError as error:
                if error.response['Error']['Code'] != 'InvalidVolume.NotFound':
                    raise
                result['deleted'] = True
                break
            time.sleep(2)
        else:
            raise RuntimeError('Volume deletion unconfirmed by read-back')
    return result
