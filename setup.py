try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Learner project',
    'author': 'ICPE Machine Learning Workgroup',
    'url': 'http://github.com/frbl/ICPE_machine_learning_workgroup',
    'download_url': 'http://github.com/frbl/ICPE_machine_learning_workgroup',
    'author_email': 'f.j.blaauw@umcg.nl',
    'version': '0.0.1',
    'install_requires': ['nose'],
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

setup(**config)

