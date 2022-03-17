# # in main folder
# rm dist/* 
# python setup.py bdist_wheel --universal

# # on some test conda env
# twine upload --repository-url https://test.pypi.org/legacy/ dist/trialML*
# pip uninstall trialML
# pip install --index-url https://test.pypi.org/simple/ trialML --user

# # Upload to PyPI: https://pypi.org/project/trialML/
# twine upload dist/trialML*
# pip uninstall trialML
# pip install trialML
