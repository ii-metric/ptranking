#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Description

"""

from org.archive.ltr_adversarial_learning.base.ad_player import AdversarialPlayer

class Pair_Generator(AdversarialPlayer):
    '''
    A pairwise generator
    '''
    def __init__(self, sf_para_dict=None):
        super(Pair_Generator, self).__init__(sf_para_dict=sf_para_dict)