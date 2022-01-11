'''
Test for translate_coords.py

Written: Joe Watchwell, December 9th 2021
'''
import sys
sys.path.append('../')
sys.path.append('../model/')
import numpy as np
import unittest
from dj_util import *


parsed_pdb = np.load('input/2KL8_parsed.npy', allow_pickle=True).item()
xyz=np.copy(parsed_pdb['xyz'])
test_list = [('A3',4.0),('A10',0.0),('A15',2.0)]
class TestTranslateCoords(unittest.TestCase):
    def test_translate_coords(self):
        test=parsed_pdb
        test_output, _ = translate_coords(parsed_pdb, test_list)
        self.assertAlmostEqual(test_output[0,0,0],xyz[0,0,0])
        self.assertNotAlmostEqual(test_output[2,0,0],xyz[2,0,0])
        self.assertAlmostEqual(test_output[9,0,0],xyz[9,0,0])
        
        temp_less = []
        temp_more = []
        i=0
        while i<1000:
            test_output, _ = translate_coords(parsed_pdb,test_list)
            x2_0 = xyz[2,0,0]
            y2_0 = xyz[2,0,1]
            z2_0 = xyz[2,0,2]
            x2_1 = test_output[2,0,0]
            y2_1 = test_output[2,0,1]
            z2_1 = test_output[2,0,2]
            x14_0 = xyz[14,0,0]
            y14_0 = xyz[14,0,1]
            z14_0 = xyz[14,0,2]
            x14_1 = test_output[14,0,0]
            y14_1 = test_output[14,0,1]
            z14_1 = test_output[14,0,2]
            temp_more.append(np.sqrt((x2_0-x2_1)**2 + (y2_0-y2_1)**2 + (z2_0-z2_1)**2))
            temp_less.append(np.sqrt((x14_0-x14_1)**2 + (y14_0-y14_1)**2 + (z14_0-z14_1)**2))
            i+=1
            test_output=0
        self.assertFalse(all(temp_more) > 2.0) 
        self.assertTrue(all(temp_less) < 2.001)
       
