from Encoders.basic_encoder import *
from AttnDecoders.attn_decoder import *
from train import *
from evaluation import *
from visualize import *
import matplotlib.pyplot as plt
import pickle
plt.switch_backend('agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    hidden_size = 256
    print("Building Encoder...")
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    print("Building Decoder...")
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    print("Starting Training...")
    trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
    print("Finished Training...")
    # Evaluation and Visualization

    print("Evaluating Rabdomly...")
    evaluateRandomly(encoder1, attn_decoder1)

    print("Saving Encoder...")
    pickle.dump(encoder1, open('Models/encV1.pkl', 'wb'))
    print("Saving Decoder...")
    pickle.dump(attn_decoder1, open('Models/adecV1.pkl', 'wb'))
    print("Saved model successfully...")

    print("Evaluating on a sentence...")
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, "how are you today? .")
    plt.matshow(attentions.numpy())

    evaluateAndShowAttention(encoder1, attn_decoder1, "I would like to have some coffee")

    evaluateAndShowAttention(encoder1, attn_decoder1, "what is the purpose of life")

    evaluateAndShowAttention(encoder1, attn_decoder1, "what is artificial intelligence")

    evaluateAndShowAttention(encoder1, attn_decoder1, "i love you")

    return


if __name__ == "__main__":
    main()




