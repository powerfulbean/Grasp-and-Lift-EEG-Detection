# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 22:43:31 2020

@author: Jin Dou
"""


from StellarBrainwav.Helper.StageControl import decrtr_stage,configStage

configStage([1,2,3])

@decrtr_stage(1)
def Add(a,b):
    return a+b



p = Add(1,3)
print(p)




