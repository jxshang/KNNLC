"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'cifar-10', 'stl-10', 'tiny-imagenet', 'cifar-100', 'imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200'}
        assert(database in db_names)

        if database == 'cifar-10':
            return '../data/cifar10/'
        
        elif database == 'cifar-100':
            return '../data/cifar100/'

        elif database == 'stl-10':
            return '../data/stl10/'
        elif database == 'tiny-imagenet':
            return '../data/tiny-imagenet/'
        elif database in ['imagenet', 'imagenet_50', 'imagenet_100', 'imagenet_200']:
            return '../data/imagenet/'
        
        else:
            raise NotImplementedError
