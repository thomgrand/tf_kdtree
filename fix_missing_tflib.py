import sys
import os
import tensorflow as tf
from os import listdir
from os.path import isfile, join

lib_dir = tf.sysconfig.get_lib()
target_file = lib_dir + "/libtensorflow_framework.so"
if not isfile(target_file):
    files = [f for f in listdir(lib_dir) if isfile(join(lib_dir, f))]

    found = False
    for src_file in files:
        if "libtensorflow_framework.so" in src_file:
            os.symlink(src_file, target_file)
            found = True
            break

    if not found:
        raise Exception("Could not find libtensorflow_framework.so, needed for the installation")