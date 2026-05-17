import os
import sys
import argparse

from ATSimCore import ATSimKernel

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lcfFile', type=str, required=True, help='lcfFile')  
    parser.add_argument('--ConfigFile', type=str, required=True, help='ConfigFile') 
    parser.add_argument('--SimParamsFile', type=str, required=True, help='SimParamsFile')  
    args = parser.parse_args()
    ATSimKernel(args)