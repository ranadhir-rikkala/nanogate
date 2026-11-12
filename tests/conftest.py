import pytest

from nanogate.db import Database
from nanogate.models import Identity


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def it_user():
    return Identity(tenant="acme", department="it", role="app", key_id="k_it")


@pytest.fixture
def hr_user():
    return Identity(tenant="acme", department="hr", role="app", key_id="k_hr")
