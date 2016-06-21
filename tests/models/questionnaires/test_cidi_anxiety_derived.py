import inspect

from learner.models.questionnaires.cidi_anxiety_derived import CIDIAnxietyDerived
import pytest
import numpy as np


class TestCIDIAnxietyDerived:
    name = 'cidi_anxiety_derived'
    filename = 'cidi.csv'
    measurement_moment = 'c'

    @pytest.fixture()
    def subject(self, mock_reader):
        subject = CIDIAnxietyDerived(name=self.name,
                                     filename=self.filename,
                                     measurement_moment=self.measurement_moment,
                                     reader=mock_reader)
        return subject

    def test_init(self, subject):
        # Test if the super class is called with the correct parameters
        assert subject.name == self.name
        assert subject.filename == self.filename
        assert subject.measurement_moment == self.measurement_moment

    def test_correct_function_mapping(self, subject):
        result = subject.function_mapping
        assert result is not None

        # Test if all exported functions are defined
        all_functions = inspect.getmembers(subject, predicate=inspect.ismethod)
        all_functions = list(map(lambda function: function[1], all_functions))
        for funckey in result.keys():
            current = result[funckey]
            assert current in all_functions

    ################
    # Social fobia #
    ################
    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy01',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy01',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy01',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_social_fobia_past_month(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.social_fobia_past_month('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy06',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy06',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy06',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_social_fobia_past_six_months(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.social_fobia_past_six_months('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy11',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy11',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy11',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_social_fobia_past_year(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.social_fobia_past_year('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy16',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy16',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy16',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_social_fobia_in_lifetime(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.social_fobia_in_lifetime('participant'), expected)

    # Panic with AgoraFobia
    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy02',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy02',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy02',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_panic_with_agora_fobia_past_month(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.panic_with_agora_fobia_past_month('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy07',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy07',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy07',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_panic_with_agora_fobia_past_six_months(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.panic_with_agora_fobia_past_six_months('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy12',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy12',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy12',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_panic_with_agora_fobia_past_year(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.panic_with_agora_fobia_past_year('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy17',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy17',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy17',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_panic_with_agora_fobia_in_lifetime(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.panic_with_agora_fobia_in_lifetime('participant'), expected)

    ############################
    # Panic without AgoraFobia #
    ############################
    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy08',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy08',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy08',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_panic_without_agora_fobia_past_six_months(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.panic_without_agora_fobia_past_six_months('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy03',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy03',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy03',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_panic_without_agora_fobia_past_month(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.panic_without_agora_fobia_past_month('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy13',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy13',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy13',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_panic_without_agora_fobia_past_year(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.panic_without_agora_fobia_past_year('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy18',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy18',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy18',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_panic_without_agora_fobia_in_lifetime(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.panic_without_agora_fobia_in_lifetime('participant'), expected)

    ##############
    # AgoraFobia #
    ##############
    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy04',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy04',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy04',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_agora_fobia_past_month(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.agora_fobia_past_month('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy09',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy09',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy09',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_agora_fobia_past_six_months(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.agora_fobia_past_six_months('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy14',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy14',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy14',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_agora_fobia_past_year(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.agora_fobia_past_year('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy19',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy19',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy19',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_agora_fobia_in_lifetime(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.agora_fobia_in_lifetime('participant'), expected)

    #######################################
    # Panic with General Anxiety Disorder #
    #######################################
    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy05',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy05',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy05',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def general_anxiety_disorder_past_month(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.general_anxiety_disorder_past_month('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy10',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy10',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy10',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_general_anxiety_disorder_past_six_months(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.general_anxiety_disorder_past_six_months('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy15',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy15',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy15',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_general_anxiety_disorder_past_year(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.general_anxiety_disorder_past_year('participant'), expected)

    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy20',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy20',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy20',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_general_anxiety_disorder_in_lifetime(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.general_anxiety_disorder_in_lifetime('participant'), expected)

    #######################################################
    # Number of current anxiety disorders (pastSixMonths) #
    #######################################################
    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy21',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy21',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy21',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_number_of_current_anxiety_diagnoses(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.number_of_current_anxiety_diagnoses('participant'), expected)

    ######################
    # Lifetime Anxiety D #
    ######################
    @pytest.mark.parametrize('mock_get_field,expected', [
        ({'participant': 'participant',
          'field': 'anxy22',
          'value': 0}, 0),
        ({'participant': 'participant',
          'field': 'anxy22',
          'value': 1}, 1),
        ({'participant': 'participant',
          'field': 'anxy22',
          'value': -1}, np.nan),
    ],
                             indirect=True)
    def test_lifetime_anxiety_diagnoses_present(self, mock_get_field, expected, assert_with_nan, subject):
        assert_with_nan(subject.lifetime_anxiety_diagnoses_present('participant'), expected)
