import argparse

from detector.toxic_detector import Detector

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model', type=str, default='AttnLSTM')
    args = parser.parse_args()

    test_text = args.text

    device = args.device
    model = args.model
    model_path = None
    
    if model == 'AttnLSTM':
        print("Using Attention-based LSTM model")
        model_path='model/Attention_based_LSTM_best_state_full.pt.pt'
    else:
        raise ValueError(f"Model {model} not supported")

    detector = Detector(model_path=model_path, vocab_path='data/vocab.txt')
    detector.set_device(device)
    output = detector(test_text)
    if(output <= 0.5):
        print("Text isn't toxic. Toxicness probability: {:.3f}".format(output.item()))
    else:
        print("Text is toxic. Toxicness probability: {:.3f}".format(output.item()))

if __name__ == '__main__':
    main()
