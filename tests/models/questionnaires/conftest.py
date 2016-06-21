import pytest


@pytest.fixture
def mock_get_field(request, subject, monkeypatch):
    my_participant = request.param['participant']
    field_name = request.param['field']
    val = request.param['value']

    def fake_get_field(participant, field):
        assert field_name == field
        assert my_participant == participant
        return val

    monkeypatch.setattr(subject, 'get_field', fake_get_field)
