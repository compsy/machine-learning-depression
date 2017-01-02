import pytest

from learner.caching.object_cacher import ObjectCacher


class TestObjectCacher():

    @pytest.fixture()
    def subject(self):
        subject = ObjectCacher(directory='tests/')
        return subject

    def test_init_can_add_directory(self):
        subject = ObjectCacher()
        assert subject.directory == 'cache/'
        subject = ObjectCacher('somethingelse')

        assert subject.directory == 'somethingelse'

    def test_raises_when_no_file_found(self, subject):
        cache_name = 'something_not_existing'
        with pytest.raises(FileNotFoundError) as err:
            subject.read_cache(cache_name)

        assert str(err.value) == 'File: tests/' + cache_name + ' not found!'

    def test_can_write_test_data(self, subject):
        data = {'a':1}
        subject.write_cache(data, 'data_examples/test_cache.pkl')
        assert subject.file_available('data_examples/test_cache.pkl')

    def test_can_read_test_data(self, subject):
        data = {'a':1}
        cache_file = 'data_examples/test_cache.pkl'
        subject.write_cache(data, cache_file)
        result = subject.read_cache(cache_file)
        assert result == data
        assert result['a'] == 1

    def test_available_can_check_if_a_file_exists(self, subject):
        cache_file = 'data_examples/test_cache.pkl'
        assert subject.file_available(cache_file)

        cache_file = 'data_examples/test_cache12313kiofjaosf'
        assert not subject.file_available(cache_file)

    def test_available_can_check_with_or_without_adding_a_dir(self, subject):
        cache_file = 'data_examples/test_cache.pkl'
        assert subject.file_available(cache_file)
        assert not subject.file_available(cache_file, add_dir=False)

        cache_file = 'tests/data_examples/test_cache.pkl'
        assert subject.file_available(cache_file, add_dir=False)