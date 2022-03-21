from setuptools import setup
from setuptools.command.install import install as _install


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
    install_requires=['nltk>=3.7', 'numpy>=1.22.3', 'requests>=2.27.1', 'scikit-learn>=1.0.2',
                      'nmf @ git+https://github.com/mariolpantunes/nmf@main#egg=nmf',
                      'knee @ git+https://github.com/mariolpantunes/knee@main#egg=knee'],
    setup_requires=['nltk>=3.7']
)
