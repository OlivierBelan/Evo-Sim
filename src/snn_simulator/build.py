import pathlib
import os

# print(pathlib.Path(__file__).parent.resolve())
# print(pathlib.Path().resolve())


# "python setup.py build_ext --inplace"

def build_simulator():
    file_abolute_location = pathlib.Path(__file__).parent.resolve()
    build_state = os.system("python "+str(file_abolute_location) + "/setup.py build_ext --inplace")
    if build_state != 0:
        print("build.sh failed")
        exit(1)
    else:
        os.system("rm "+str(file_abolute_location) + "/*.so")
        print("build.sh success\n\n")

def clean_simulator():
    file_running_location = str(pathlib.Path().resolve())
    os.system("rm "+ file_running_location +"/*.so")

# build_simulator()
# clean_simulator()