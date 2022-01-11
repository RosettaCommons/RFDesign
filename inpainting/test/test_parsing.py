# Test suite associated with anything for parsing / mapping inputs
import numpy as np 
import torch 
import sys 
import os 

import unittest 

THIS_FILE = os.path.realpath(__file__)
THIS_DIR  = os.path.dirname(THIS_FILE)

print(THIS_FILE)
print(THIS_DIR)

### Package imports 
sys.path.append('../../hallucinate/util')
import parsers 

class TestMask(unittest.TestCase):

    def test_window_contig_correspondence(self):
        """
        Test coresspondence between outputs that should be the same from window mode parsing 
        and contigs parsing 
        """
    
        
        #parsed_pdb = 
        
        # contigs str 
        window1 = 'A,5,6:A,25-30'   
        window2 = 'A,70-79'

        # corrresponding window str 


class TestParser(unittest.TestCase):

    def test_parser(self):
        """
        Tests our pdb parsers 
        """
        # load / parse pdb 
        true_path   = os.path.join(THIS_DIR,'input','2KL8_parsed.npy')
        true_parsed = np.load(true_path, allow_pickle=True).item()

        test_pdb = os.path.join(THIS_DIR, 'input','2KL8.pdb')
        test_parsed = parsers.parse_pdb(test_pdb)

        ### assertions ###

        # check identical crds 
        xyz_true = true_parsed['xyz']
        xyz_test = test_parsed['xyz']
        self.assertEqual((xyz_true == xyz_test).all(), True)

        # check identical seq
        seq_true = true_parsed['seq']
        seq_test = test_parsed['seq']
        self.assertEqual((seq_true == seq_test).all(), True)

        # indices 
        #self.assertIs( test_parsed['idx'][0],  ('A',1))
        #self.assertIs( test_parsed['idx'][-1], ('A',79))
         
        



        
          




