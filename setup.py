#TODO: update to better framework
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
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger_eng')


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
    install_requires=['nltk>=3.9.1', 'numpy>=1.26.4', 'requests>=2.32.3', 'scikit-learn>=1.5.2', 
    'joblib>=1.4.2', 'pyNNMF>=0.1.3', 'kneeliverse>=1.0', 'numba>=0.61.0'],
    setup_requires=['nltk>=3.9.1']
)
