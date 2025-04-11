import argparse

from detector.toxic_detector import Detector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)
    args = parser.parse_args()

    test_text = args.text

    detector = Detector(model_path='model/QuoraLSTM_best_state_full.pt', vocab_path='data/vocab.txt')
    detector.set_device('cuda:0')
    output = detector(test_text)
    if(output <= 0.5):
        print("Text isn't toxic. Toxicness probability: {:.3f}".format(output.item()))
    else:
        print("Text is toxic. Toxicness probability: {:.3f}".format(output.item()))

if __name__ == '__main__':
    main()
