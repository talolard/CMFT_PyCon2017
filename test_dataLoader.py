from unittest import TestCase
import sys
from arg_parse import parse_args
from prepare_data import DataLoader


class TestDataLoader(TestCase):
    def setUp(self):
        self.args = parse_args(sys.argv[3:])
        self.args.data_path='data/test/input.txt'
        self.args.saved_data_path = 'data/test/dataset.npa'
        self.DL = DataLoader(self.args)
    def test_load_txt_data(self):
        org_low = self.DL.load_txt_data()

    def test_txt_to_array(self):
        org_lower = self.DL.load_txt_data()
        dataset = self.DL.txt_to_array(org_lower)
        self.assertEqual(dataset.shape,(len(org_lower),2,self.DL.max_len))

    def test_load_data(self):
        self.DL.load_data()

    def test_get_batch(self):
        self.DL.load_data()
        for num,(orig,low) in enumerate(self.DL.get_batch()):
            self.assertLess(num,1000)
        print(''.join(map(chr,orig[0])))
        print(''.join(map(chr, low[0])))

