import inspect
from learner.data_output.std_logger import L
import pytest
import numpy as np

import logging

class TestL():
    def test_csvexporter_exposes_a_static_function(self):
        static_methods =list(map(lambda a: a[0], inspect.getmembers(L, predicate=inspect.isfunction)))
        assert 'setup' in static_methods
        assert 'info' in static_methods
        assert 'br' in static_methods
        assert 'debug' in static_methods
        assert 'warn' in static_methods

    def test_setup_creates_a_new_output_log_file(self, monkeypatch):
        # I'm not sure how to test this, as all functions are static and it seems I'm not able to mock those
        pass

    @pytest.mark.skip(reason="The global var is defined in the L module, realy weird, so this spec fails. Probably we should convert it to a singleton (the logger)")
    def test_info_calls_setup_if_not_called(self, monkeypatch):
        def fake_setup(logger):
            assert logger == False
            raise ValueError('stop_execution')

        monkeypatch.setattr(L, 'setup', fake_setup)

        with pytest.raises(ValueError, message='stop_execution') as err:
            globals().pop('logger_on_hpc', None)
            L.info('jfadslkjdfaslkfjlsakjfdklsjdf')
        assert str(err.value) == 'stop_execution'

    def test_info_prints_only_on_rank_0(self, monkeypatch):
        # Also hard to test, as we only have static funcs
        pass
