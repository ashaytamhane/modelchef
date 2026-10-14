from types import SimpleNamespace

import pytest

from modelchef.aws_storage import manage_volume


def test_ebs_deletion_is_explicit_and_verified():
    from botocore.exceptions import ClientError
    volume_id = 'vol-0123456789abcdef0'
    calls = []
    volume = {'VolumeId': volume_id, 'State': 'available', 'Attachments': []}
    def describe(**kw):
        if any(kind == 'delete' for kind, _ in calls):
            calls.append(('verify', kw))
            raise ClientError({'Error': {'Code': 'InvalidVolume.NotFound'}}, 'DescribeVolumes')
        return {'Volumes': [volume]}
    ec2 = SimpleNamespace(describe_volumes=describe,
                         delete_volume=lambda **kw: calls.append(('delete', kw)))
    assert not manage_volume(volume_id, 'us-east-1', 'delete', False, ec2=ec2)['deleted']
    assert not manage_volume(volume_id, 'us-east-1', 'status', True, ec2=ec2)['deleted']
    assert calls == []
    assert manage_volume(volume_id, 'us-east-1', 'delete', True, ec2=ec2)['deleted']
    assert [kind for kind, _ in calls] == ['delete', 'verify']
    calls.clear()
    volume['Attachments'] = [{'InstanceId': 'i-test'}]
    with pytest.raises(ValueError, match='attached'):
        manage_volume(volume_id, 'us-east-1', 'delete', True, ec2=ec2)
