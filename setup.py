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
    'packages': ['learner'],
    'scripts': [],
    'name': 'learner'
}

setup(**config)
