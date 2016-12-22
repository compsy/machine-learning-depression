# setup.py
import os
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

config = {
    'description': 'Learner project',
    'author': 'ICPE Machine Learning Workgroup',
    'url': 'http://github.com/frbl/ICPE_machine_learning_workgroup',
    'download_url': 'http://github.com/frbl/ICPE_machine_learning_workgroup',
    'author_email': 'f.j.blaauw@umcg.nl',
    'version': '0.0.1',
    'install_requires': requirements,
    'packages': ['learner',
                 'learner.data_input',
                 'learner.data_output',
                 'learner.data_output.plotters',
                 'learner.data_transformers',
                 'learner.factories',
                 'learner.machine_learning_evaluation',
                 'learner.machine_learning_models',
                 'learner.machine_learning_models.model_runners',
                 'learner.machine_learning_models.models',
                 'learner.models',
                 'learner.models.questionnaires',
                 'learner.output_file_creators'],
    'scripts': [],
    'setup_requires': ['pytest-runner'],
    'tests_require': ['pytest', 'sphinx'],
    'name': 'learner'
}

if sys.argv[-1] == 'test':
    test_requirements = [
        'pytest',
        'flake8',
        'coverage'
    ]
    try:
        modules = map(__import__, test_requirements)
    except ImportError as e:
        err_msg = e.message.replace("No module named ", "")
        msg = "%s is not installed. Install your test requirments." % err_msg
        raise ImportError(msg)
    os.system('py.test')
    sys.exit()

setup(**config)

