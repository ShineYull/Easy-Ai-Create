import os
import torch
import argparse

from src.modules.ui.ui_manage import UIManage

def parse_args(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mac",
        action='store_true',
        help="Use bfloat16",
    )
    
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    uiManage = UIManage()
    uiManage.ui_full()