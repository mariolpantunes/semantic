import pathlib
from setuptools import setup, find_packages
from setuptools.command.install import install as _install


here = pathlib.Path(__file__).parent.resolve()
long_description = (here/'README.md').read_text(encoding='utf-8')


class DownloadNLTK(_install):
    def run(self):
        self.do_egg_install()
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')


setup(
    name='semantic',
    version='1.0',
    description='DPW and DPWC semantic models',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Mário antunes',
    author_email='mariolpantunes@gmail.com',
    url='https://github.com/mariolpantunes/semantic',
    packages=find_packages(),
    cmdclass={'download_nltk': DownloadNLTK},
    install_requires=['nltk>=3.8.1', 'numpy>=1.26.4', 'requests>=2.32.3', 'scikit-learn>=1.5.0', 'joblib>=1.4.2', 'pyNNMF>=0.1.3', 'kneeliverse>=1.0'],
    setup_requires=['nltk>=3.8.1']
)
