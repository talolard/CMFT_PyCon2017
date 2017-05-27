import argparse

def parse_args(args):
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--data_path', type=str,
                        default='data/wikipedia/input.txt', help="Path to raw txt file")
    parser.add_argument('--saved_data_path', type=str,
                        default='data/wikipedia/dataset.npa', help="Where to save/load processed input")
    parser.add_argument('--batch_size', type=int,
                        default=8, help="Number of sentences in batch")
    parser.add_argument('--lr', type=int,
                        default=0.0001, help="Learning Rate")

    # ...Create your parser as you like...
    return parser.parse_args(args)
