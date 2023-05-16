from setuptools import setup, find_packages


setup(name='Purification',
      version='1.0.0',
      description='Learning to Purification for Unsupervised Person Re-identification',
      author='Xiao Teng',
      author_email='tengxiao14@nudt.edu.cn',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Person Re-identification'
      ])
