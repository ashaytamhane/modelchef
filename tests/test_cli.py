from ashayllm.cli import check_environment


def test_environment_shape():
    info = check_environment()
    assert "python" in info
    assert isinstance(info["cuda_available"], bool)
