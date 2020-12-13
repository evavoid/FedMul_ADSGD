"""
-----------------------------------------------------------------------------------
--  @file       MLFoglib->options_base.py
--  @author     Ma Haoming(En:louis)(https://github.com/evavoid)
--  @brief      xxxxxxxxx
    
--  @Ide        PyCharm
--  @time       2020/12/12-21:45
-----------------------------------------------------------------------------------
"""
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusers_num', type=int, default=5, help="user num of fmnist dataset")
    parser.add_argument('--musers_num', type=int, default=5, help="user num of mnist dataset")
    parser.add_argument('--epochs', type=int, default=20, help="rounds of training")
    parser.add_argument('--bs', type=int, default=200, help="test batch size")
    parser.add_argument('--lr', type=int, default=0.01, help="learning_rate")
    args = parser.parse_args()
    return args
