from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='Demo_Project_CNN_digits_recognition_Benbhk',
      version="1.0",
      description="This project is a demonstration. It uses a convolutional neural network (CNN) to recognize a number. The interface is coded with Streamlit (framework). The project is available online on two platforms:",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/Demo_Project_CNN_digits_recognition_Benbhk-run'],
      zip_safe=False)
