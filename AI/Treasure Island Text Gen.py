{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "jppuvXtnjZOY"
      },
      "source": [
        "**Chapter 16 – Natural Language Processing with RNNs and Attention**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "h88VuExPjZOb"
      },
      "source": [
        "_This notebook contains all the sample code and solutions to the exercises in chapter 16._"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y7KhVHbOjZOc"
      },
      "source": [
        "<table align=\"left\">\n",
        "  <td>\n",
        "    <a href=\"https://colab.research.google.com/github/ageron/handson-ml3/blob/main/16_nlp_with_rnns_and_attention.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
        "  </td>\n",
        "  <td>\n",
        "    <a target=\"_blank\" href=\"https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/16_nlp_with_rnns_and_attention.ipynb\"><img src=\"https://kaggle.com/static/images/open-in-kaggle.svg\" /></a>\n",
        "  </td>\n",
        "</table>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dFXIv9qNpKzt",
        "tags": []
      },
      "source": [
        "# Setup"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "8IPbJEmZpKzu"
      },
      "source": [
        "This project requires Python 3.7 or above:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TFSU3FCOpKzu"
      },
      "outputs": [],
      "source": [
        "import sys\n",
        "\n",
        "assert sys.version_info >= (3, 7)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "GJtVEqxfpKzw"
      },
      "source": [
        "And TensorFlow ≥ 2.8:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0Piq5se2pKzx"
      },
      "outputs": [],
      "source": [
        "from packaging import version\n",
        "import tensorflow as tf\n",
        "\n",
        "assert version.parse(tf.__version__) >= version.parse(\"2.8.0\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DDaDoLQTpKzx"
      },
      "source": [
        "As we did in earlier chapters, let's define the default font sizes to make the figures prettier:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8d4TH3NbpKzx"
      },
      "outputs": [],
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "plt.rc('font', size=14)\n",
        "plt.rc('axes', labelsize=14, titlesize=14)\n",
        "plt.rc('legend', fontsize=14)\n",
        "plt.rc('xtick', labelsize=10)\n",
        "plt.rc('ytick', labelsize=10)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RcoUIRsvpKzy"
      },
      "source": [
        "And let's create the `images/nlp` folder (if it doesn't already exist), and define the `save_fig()` function which is used through this notebook to save the figures in high-res for the book:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "PQFH5Y9PpKzy"
      },
      "outputs": [],
      "source": [
        "from pathlib import Path\n",
        "\n",
        "IMAGES_PATH = Path() / \"images\" / \"nlp\"\n",
        "IMAGES_PATH.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "def save_fig(fig_id, tight_layout=True, fig_extension=\"png\", resolution=300):\n",
        "    path = IMAGES_PATH / f\"{fig_id}.{fig_extension}\"\n",
        "    if tight_layout:\n",
        "        plt.tight_layout()\n",
        "    plt.savefig(path, format=fig_extension, dpi=resolution)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YTsawKlapKzy"
      },
      "source": [
        "This chapter can be very slow without a GPU, so let's make sure there's one, or else issue a warning:"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Ekxzo6pOpKzy"
      },
      "outputs": [],
      "source": [
        "if not tf.config.list_physical_devices('GPU'):\n",
        "    print(\"No GPU was detected. Neural nets can be very slow without a GPU.\")\n",
        "    if \"google.colab\" in sys.modules:\n",
        "        print(\"Go to Runtime > Change runtime and select a GPU hardware \"\n",
        "              \"accelerator.\")\n",
        "    if \"kaggle_secrets\" in sys.modules:\n",
        "        print(\"Go to Settings > Accelerator and select GPU.\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "83Ow6FFvjZOg"
      },
      "source": [
        "# Generating Treasure Island text Using a Character RNN"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iRlL6J6FjZOg"
      },
      "source": [
        "## Creating the Training Dataset"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7QB4Bjf4rkav",
        "outputId": "971e753e-00b6-413f-8098-6b16b7b5a826"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "q9j7y3upjZOi"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "\n",
        "filepath = \"/content/drive/MyDrive/Colab Notebooks/Advanced AI/TreasureIslandClean.txt\"\n",
        "with open(filepath) as f:\n",
        "    island_text = f.read()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TfnNHSZ5jZOo",
        "outputId": "0361b188-f4ad-4583-bb24-127ecba80a94",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "TREASURE ISLAND\n",
            "\n",
            "by Robert Louis Stevenson\n",
            "\n",
            "\n",
            "\n",
            "\n",
            "TREASURE ISLAND\n",
            "\n",
            "To S.L.O., an Am\n"
          ]
        }
      ],
      "source": [
        "# extra code – shows a short text sample\n",
        "print(island_text[:80])"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "default encoding is word level. we will do character level encoding. we use standardize=\"lower\" to convert the text to lowercase (which will simplify the task):"
      ],
      "metadata": {
        "id": "W4RQnFVpzTXi"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "*text is converted to vectors and and each char is given an int. Clean dataset is encoded and ready for AI model.*"
      ],
      "metadata": {
        "id": "SIQptsAZfECZ"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "uxHxhRHFjZOp"
      },
      "outputs": [],
      "source": [
        "text_vec_layer = tf.keras.layers.TextVectorization(split=\"character\",\n",
        "                                                   standardize=\"lower\")\n",
        "text_vec_layer.adapt([shakespeare_text])\n",
        "encoded = text_vec_layer([shakespeare_text])[0]"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Each character is now mapped to an integer, starting at 2. The TextVectorization layer reserved the value 0 for padding tokens, and it reserved 1 for unknown characters. We won’t need either of these tokens for now, so let’s subtract 2 from the character IDs and compute the number of distinct characters and the total number of characters."
      ],
      "metadata": {
        "id": "ts7RGI_UzsiJ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "print(encoded)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "XIscAz4KsYE-",
        "outputId": "631d5898-1462-4e05-b8d9-6ecb639262d2"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tf.Tensor([ 4 11  3 ... 34 31 15], shape=(364309,), dtype=int64)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "encoded.shape"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "da-KCH-Ls71R",
        "outputId": "d3e0ca0d-aca9-47f7-f53a-eb42938eda35"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "TensorShape([364309])"
            ]
          },
          "metadata": {},
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ucAW9dnSjZOp"
      },
      "outputs": [],
      "source": [
        "encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use\n",
        "n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39\n",
        "dataset_size = len(encoded)  # total number of chars = 1,115,394"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "glRFvuntjZOq",
        "outputId": "9424cbdb-80d9-481e-a8cc-09e3812bc51c",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "55"
            ]
          },
          "metadata": {},
          "execution_count": 14
        }
      ],
      "source": [
        "n_tokens"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eLQllfHcjZOq",
        "outputId": "5564d30a-27fb-498b-ce4e-989ce0aba9fd",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "364309"
            ]
          },
          "metadata": {},
          "execution_count": 15
        }
      ],
      "source": [
        "dataset_size"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "we can turn this very long sequence into a dataset of windows that we can then use to train a sequence-to-sequence RNN. The targets will be similar to the inputs, but shifted by one time step into the “future”. For example, one sample in the dataset may be a sequence of character IDs representing the text “to be or not to b” (without the final “e”), and the corresponding target— a sequence of character IDs representing the text “o be or not to be” (with the final “e”, but without the leading “t”). Let’s write a small utility function to convert a long sequence of character IDs into a dataset of input/target window pairs.\n",
        "\n",
        "It takes a sequence as input (i.e., the encoded text), and creates a dataset containing all the windows of the desired length. • It increases the length by one, since we need the next character for the target. • Then, it shuffles the windows (optionally), batches them, splits them into input/ output pairs, and activates prefetching."
      ],
      "metadata": {
        "id": "IZxHE9C20be3"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-T1LdhkBjZOq"
      },
      "outputs": [],
      "source": [
        "def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):\n",
        "    ds = tf.data.Dataset.from_tensor_slices(sequence)\n",
        "    ds = ds.window(length + 1, shift=1, drop_remainder=True)\n",
        "    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))\n",
        "    if shuffle:\n",
        "        ds = ds.shuffle(100_000, seed=seed)\n",
        "    ds = ds.batch(batch_size)\n",
        "    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "cHYlPwCtjZOq",
        "outputId": "a6737fd7-73cb-4fc4-9e82-868db23cfedf",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[(<tf.Tensor: shape=(1, 4), dtype=int64, numpy=array([[ 4,  6,  2, 23]])>,\n",
              "  <tf.Tensor: shape=(1, 4), dtype=int64, numpy=array([[ 6,  2, 23,  3]])>)]"
            ]
          },
          "metadata": {},
          "execution_count": 17
        }
      ],
      "source": [
        "# extra code – a simple example using to_dataset()\n",
        "# There's just one sample in this dataset: the input represents \"to b\" and the\n",
        "# output represents \"o be\"\n",
        "list(to_dataset(text_vec_layer([\"To be\"])[0], length=4))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "We will create the training set, the validation set, and the test set. We will use roughly 90% of the text for training, 5% for validation, and 5% for testing.\n",
        "\n",
        "\n",
        "We set the window length to 100, but you can try tuning it: it’s easier and faster to train RNNs on shorter input sequences, but the RNN will not be able to learn any pattern longer than length , so don’t make it too small."
      ],
      "metadata": {
        "id": "6T7al8we1xVL"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_MscxtegjZOq"
      },
      "outputs": [],
      "source": [
        "length = 100\n",
        "tf.random.set_seed(42)\n",
        "train_set = to_dataset(encoded[:1_000_000], length=length, shuffle=True,\n",
        "                       seed=42)\n",
        "valid_set = to_dataset(encoded[1_000_000:1_060_000], length=length)\n",
        "test_set = to_dataset(encoded[1_060_000:], length=length)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vorS8b4VjZOr"
      },
      "source": [
        "## Building and Training the Char-RNN Model"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zhBn_dILjZOr"
      },
      "source": [
        "**Warning**: the following code may one or two hours to run, depending on your GPU. Without a GPU, it may take over 24 hours. If you don't want to wait, just skip the next two code cells and run the code below to download a pretrained model."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vivIkRsxjZOr"
      },
      "source": [
        "**Note**: the `GRU` class will only use cuDNN acceleration (assuming you have a GPU) when using the default values for the following arguments: `activation`, `recurrent_activation`, `recurrent_dropout`, `unroll`, `use_bias` and `reset_after`."
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "We use an Embedding layer as the first layer, to encode the character IDs. The Embedding layer’s number of input dimensions is the number of distinct character IDs, and the number of output dimensions is a hyperparameter you can tune— we’ll set it to 16 for now. Whereas the inputs of the Embedding layer will be 2D tensors of shape [ batch size , window length ], the output of the Embedding layer will be a 3D tensor of shape [ batch size , window length , embedding size ]. • We use a Dense layer for the output layer: it must have 39 units ( n_tokens ) because there are 39 distinct characters in the text, and we want to output a probability for each possible character (at each time step). The 39 output probabilities should sum up to 1 at each time step, so we apply the softmax activation function to the outputs of the Dense layer. • Lastly, we compile this model, using the \"sparse_categorical_crossentropy\" loss and a Nadam optimizer, and we train the model for several epochs, 3 using a ModelCheckpoint callback to save the best model (in terms of validation accuracy) as training progresses."
      ],
      "metadata": {
        "id": "x5I6QyIU2yoq"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "*last layer is prediction layer. in deep machine learning las layers number of neurons = number of possible predictions. for char prediction, char should be one of our 39 distinct char's. n_tokens = 39 for each distinct char in the training text. middle layers are hiden layers. if accuracy is bad add more layers. First layer is embedding layer where windows are fed.*"
      ],
      "metadata": {
        "id": "lkNpRiv7mIiU"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DXNuiaAxjZOr",
        "outputId": "0d173e0f-4c55-41fb-adb1-917261ec81ec",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "  11381/Unknown - 168s 13ms/step - loss: 1.4936 - accuracy: 0.5518"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 168s 13ms/step - loss: 1.4936 - accuracy: 0.5519\n",
            "Epoch 2/10\n",
            "11382/11382 [==============================] - ETA: 0s - loss: 1.2691 - accuracy: 0.6102"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 152s 12ms/step - loss: 1.2691 - accuracy: 0.6102\n",
            "Epoch 3/10\n",
            "11380/11382 [============================>.] - ETA: 0s - loss: 1.2307 - accuracy: 0.6206"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 153s 12ms/step - loss: 1.2307 - accuracy: 0.6206\n",
            "Epoch 4/10\n",
            "11381/11382 [============================>.] - ETA: 0s - loss: 1.2124 - accuracy: 0.6254"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 149s 12ms/step - loss: 1.2124 - accuracy: 0.6254\n",
            "Epoch 5/10\n",
            "11380/11382 [============================>.] - ETA: 0s - loss: 1.2009 - accuracy: 0.6285"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 148s 11ms/step - loss: 1.2009 - accuracy: 0.6285\n",
            "Epoch 6/10\n",
            "11382/11382 [==============================] - ETA: 0s - loss: 1.1931 - accuracy: 0.6305"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 146s 11ms/step - loss: 1.1931 - accuracy: 0.6305\n",
            "Epoch 7/10\n",
            "11377/11382 [============================>.] - ETA: 0s - loss: 1.1873 - accuracy: 0.6320"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 146s 11ms/step - loss: 1.1873 - accuracy: 0.6320\n",
            "Epoch 8/10\n",
            "11381/11382 [============================>.] - ETA: 0s - loss: 1.1828 - accuracy: 0.6333"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 146s 11ms/step - loss: 1.1828 - accuracy: 0.6333\n",
            "Epoch 9/10\n",
            "11381/11382 [============================>.] - ETA: 0s - loss: 1.1793 - accuracy: 0.6345"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 150s 12ms/step - loss: 1.1793 - accuracy: 0.6345\n",
            "Epoch 10/10\n",
            "11380/11382 [============================>.] - ETA: 0s - loss: 1.1762 - accuracy: 0.6354"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:tensorflow:Can save best model only with val_accuracy available, skipping.\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\r11382/11382 [==============================] - 188s 15ms/step - loss: 1.1762 - accuracy: 0.6354\n"
          ]
        }
      ],
      "source": [
        "tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU\n",
        "model = tf.keras.Sequential([\n",
        "    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),\n",
        "    tf.keras.layers.GRU(128, return_sequences=True),\n",
        "    tf.keras.layers.Dense(n_tokens, activation=\"softmax\")\n",
        "])\n",
        "\n",
        "#loss function is categorical crosentropy because we're predicting 39 categories. accuracyis used for classification.\n",
        "#If we're predicting a number it, it wold be regression and loss function. Optimizer and metrics would have changed. adam is the most famous optimizer.\n",
        "\n",
        "model.compile(loss=\"sparse_categorical_crossentropy\", optimizer=\"nadam\",\n",
        "              metrics=[\"accuracy\"])\n",
        "\n",
        "#model checkpoint is used to make the model stop at highest accuracy, loss is the lowest\n",
        "model_ckpt = tf.keras.callbacks.ModelCheckpoint(\n",
        "    \"my_shakespeare_model\", monitor=\"val_accuracy\", save_best_only=True)\n",
        "\n",
        "#fit is creating the model\n",
        "#we don't have to assign it to a history variable but it is good to visualise loss and accuracy later. we use history to access thee variables.\n",
        "history = model.fit(train_set, validation_data=valid_set, epochs=10,\n",
        "                    callbacks=[model_ckpt])"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "This model does not handle text preprocessing, so let’s wrap it in a final model containing the tf.keras.layers.TextVectorization layer as the first layer, plus a tf.keras.layers.Lambda layer to subtract 2 from the character IDs since we’re not using the padding and unknown tokens for now:"
      ],
      "metadata": {
        "id": "vnuSdLXo3xwY"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nG0xIhDHjZOs"
      },
      "outputs": [],
      "source": [
        "shakespeare_model = tf.keras.Sequential([\n",
        "    text_vec_layer,\n",
        "    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens\n",
        "    model\n",
        "])"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "I8T23XtSjZOs"
      },
      "source": [
        "If you don't want to wait for training to complete, I've pretrained a model for you. The following code will download it. Uncomment the last line if you want to use it instead of the model trained above."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5g810IDejZOt"
      },
      "outputs": [],
      "source": [
        "# extra code – downloads a pretrained model\n",
        "url = \"https://github.com/ageron/data/raw/main/shakespeare_model.tgz\"\n",
        "path = tf.keras.utils.get_file(\"shakespeare_model.tgz\", url, extract=True)\n",
        "model_path = Path(path).with_name(\"shakespeare_model\")\n",
        "#shakespeare_model = tf.keras.models.load_model(model_path)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0hhz6e82jZOu",
        "outputId": "ccaaf9dc-4a43-4427-c32d-4c59e6d9e179",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 53
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 48ms/step\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'h'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 28
        }
      ],
      "source": [
        "y_proba = shakespeare_model.predict(['If sailor tales t'])[0,-1]\n",
        "###\n",
        "\n",
        "#y_proba = shakespeare_model.predict([\"To be or not to b\"])[0, -1]\n",
        "y_pred = tf.argmax(y_proba)  # choose the most probable character ID\n",
        "text_vec_layer.get_vocabulary()[y_pred + 2]"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Great, the model correctly predicted the next character. Now let’s use this model to pretend we’re Shakespeare!"
      ],
      "metadata": {
        "id": "_LnqKtnu5LK5"
      }
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UI8Lq5gljZOv"
      },
      "source": [
        "## Generating Fake Shakespearean Text"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "To generate new text using the char-RNN model, we could feed it some text, make the model predict the most likely next letter, add it to the end of the text, then give the extended text to the model to guess the next letter, and so on. This is called greedy decoding . But in practice this often leads to the same words being repeated over and over again. Instead, we can sample the next character randomly, with a probability equal to the estimated probability, using TensorFlow’s tf.random.categorical() function. This will generate more diverse and interesting text. The categorical() function samples random class indices, given the class log probabilities (logits). For example:"
      ],
      "metadata": {
        "id": "GkNmBVlz43kR"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5G38f_IvjZOv",
        "outputId": "494c32f0-b4f0-421e-a1b2-923a487af5fe",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<tf.Tensor: shape=(1, 8), dtype=int64, numpy=array([[0, 0, 1, 1, 1, 0, 0, 0]])>"
            ]
          },
          "metadata": {},
          "execution_count": 29
        }
      ],
      "source": [
        "log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%\n",
        "tf.random.set_seed(42)\n",
        "tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "To have more control over the diversity of the generated text, we can divide the logits by a number called the temperature , which we can tweak as we wish. A temperature close to zero favors high-probability characters, while a high temperature gives all characters an equal probability. Lower temperatures are typically preferred when generating fairly rigid and precise text, such as mathematical equations, while higher temperatures are preferred when generating more diverse and creative text. The following next_char() custom helper function uses this approach to pick the next character to add to the input text:"
      ],
      "metadata": {
        "id": "d1HlBxBE6Cpq"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4zsG2gsjjZOv"
      },
      "outputs": [],
      "source": [
        "def next_char(text, temperature=1):\n",
        "    y_proba = shakespeare_model.predict([text])[0, -1:]\n",
        "    rescaled_logits = tf.math.log(y_proba) / temperature\n",
        "    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]\n",
        "    return text_vec_layer.get_vocabulary()[char_id + 2]"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Next, we can write another small helper function that will to get the next character and append it to the given text:"
      ],
      "metadata": {
        "id": "evHpRMrY9j12"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1d2dnpGXjZOw"
      },
      "outputs": [],
      "source": [
        "def extend_text(text, n_chars=50, temperature=1):\n",
        "    for _ in range(n_chars):\n",
        "        text += next_char(text, temperature)\n",
        "    return text"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "WApAHMo3jZOw"
      },
      "outputs": [],
      "source": [
        "tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "D5xuumgjjZOx",
        "outputId": "a5f8b163-2c3c-4595-ee09-d86bc27450de",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 52ms/step\n",
            "1/1 [==============================] - 0s 45ms/step\n",
            "1/1 [==============================] - 0s 61ms/step\n",
            "1/1 [==============================] - 0s 115ms/step\n",
            "1/1 [==============================] - 0s 128ms/step\n",
            "1/1 [==============================] - 0s 105ms/step\n",
            "1/1 [==============================] - 0s 68ms/step\n",
            "1/1 [==============================] - 0s 99ms/step\n",
            "1/1 [==============================] - 0s 100ms/step\n",
            "1/1 [==============================] - 0s 95ms/step\n",
            "1/1 [==============================] - 0s 51ms/step\n",
            "1/1 [==============================] - 0s 141ms/step\n",
            "1/1 [==============================] - 0s 83ms/step\n",
            "1/1 [==============================] - 0s 74ms/step\n",
            "1/1 [==============================] - 0s 65ms/step\n",
            "1/1 [==============================] - 0s 76ms/step\n",
            "1/1 [==============================] - 0s 106ms/step\n",
            "1/1 [==============================] - 0s 90ms/step\n",
            "1/1 [==============================] - 0s 78ms/step\n",
            "1/1 [==============================] - 0s 52ms/step\n",
            "1/1 [==============================] - 0s 47ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 52ms/step\n",
            "1/1 [==============================] - 0s 62ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 50ms/step\n",
            "1/1 [==============================] - 0s 67ms/step\n",
            "1/1 [==============================] - 0s 51ms/step\n",
            "1/1 [==============================] - 0s 50ms/step\n",
            "1/1 [==============================] - 0s 46ms/step\n",
            "1/1 [==============================] - 0s 45ms/step\n",
            "1/1 [==============================] - 0s 45ms/step\n",
            "1/1 [==============================] - 0s 47ms/step\n",
            "1/1 [==============================] - 0s 51ms/step\n",
            "1/1 [==============================] - 0s 52ms/step\n",
            "1/1 [==============================] - 0s 56ms/step\n",
            "1/1 [==============================] - 0s 51ms/step\n",
            "1/1 [==============================] - 0s 54ms/step\n",
            "1/1 [==============================] - 0s 51ms/step\n",
            "1/1 [==============================] - 0s 53ms/step\n",
            "1/1 [==============================] - 0s 51ms/step\n",
            "1/1 [==============================] - 0s 46ms/step\n",
            "1/1 [==============================] - 0s 46ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 52ms/step\n",
            "1/1 [==============================] - 0s 50ms/step\n",
            "1/1 [==============================] - 0s 142ms/step\n",
            "If sailor tales to sailor tunes, and the ship and still stores and the captain wit\n"
          ]
        }
      ],
      "source": [
        "print(extend_text(\"If sailor tales to sailor tunes,\", temperature=0.01))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "tzB5lNeJjZOx",
        "outputId": "733f2d44-cf6b-41a2-dba1-606ff14cbb06",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 71ms/step\n",
            "1/1 [==============================] - 0s 75ms/step\n",
            "1/1 [==============================] - 0s 67ms/step\n",
            "1/1 [==============================] - 0s 60ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 46ms/step\n",
            "1/1 [==============================] - 0s 55ms/step\n",
            "1/1 [==============================] - 0s 46ms/step\n",
            "1/1 [==============================] - 0s 56ms/step\n",
            "1/1 [==============================] - 0s 56ms/step\n",
            "1/1 [==============================] - 0s 86ms/step\n",
            "1/1 [==============================] - 0s 90ms/step\n",
            "1/1 [==============================] - 0s 50ms/step\n",
            "1/1 [==============================] - 0s 54ms/step\n",
            "1/1 [==============================] - 0s 67ms/step\n",
            "1/1 [==============================] - 0s 47ms/step\n",
            "1/1 [==============================] - 0s 46ms/step\n",
            "1/1 [==============================] - 0s 45ms/step\n",
            "1/1 [==============================] - 0s 50ms/step\n",
            "1/1 [==============================] - 0s 47ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 54ms/step\n",
            "1/1 [==============================] - 0s 51ms/step\n",
            "1/1 [==============================] - 0s 55ms/step\n",
            "1/1 [==============================] - 0s 53ms/step\n",
            "1/1 [==============================] - 0s 63ms/step\n",
            "1/1 [==============================] - 0s 57ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "1/1 [==============================] - 0s 47ms/step\n",
            "1/1 [==============================] - 0s 67ms/step\n",
            "1/1 [==============================] - 0s 100ms/step\n",
            "1/1 [==============================] - 0s 78ms/step\n",
            "1/1 [==============================] - 0s 86ms/step\n",
            "1/1 [==============================] - 0s 94ms/step\n",
            "1/1 [==============================] - 0s 100ms/step\n",
            "1/1 [==============================] - 0s 95ms/step\n",
            "1/1 [==============================] - 0s 125ms/step\n",
            "1/1 [==============================] - 0s 103ms/step\n",
            "1/1 [==============================] - 0s 128ms/step\n",
            "1/1 [==============================] - 0s 68ms/step\n",
            "1/1 [==============================] - 0s 50ms/step\n",
            "1/1 [==============================] - 0s 100ms/step\n",
            "1/1 [==============================] - 0s 101ms/step\n",
            "1/1 [==============================] - 0s 131ms/step\n",
            "1/1 [==============================] - 0s 83ms/step\n",
            "1/1 [==============================] - 0s 80ms/step\n",
            "1/1 [==============================] - 0s 128ms/step\n",
            "1/1 [==============================] - 0s 78ms/step\n",
            "1/1 [==============================] - 0s 66ms/step\n",
            "1/1 [==============================] - 0s 49ms/step\n",
            "If sailor tales to sailor tunes, mate,” he souther about and shot into the rest, m\n"
          ]
        }
      ],
      "source": [
        "print(extend_text(\"If sailor tales to sailor tunes,\", temperature=1))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6FLV8qe7jZOx",
        "outputId": "9e4467a0-d9c7-46fe-ed61-012b5c26e498",
        "colab": {
          "base_uri": "https://localhost:8080/"
        }
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "1/1 [==============================] - 0s 31ms/step\n",
            "1/1 [==============================] - 0s 34ms/step\n",
            "1/1 [==============================] - 0s 29ms/step\n",
            "1/1 [==============================] - 0s 32ms/step\n",
            "1/1 [==============================] - 0s 30ms/step\n",
            "1/1 [==============================] - 0s 38ms/step\n",
            "1/1 [==============================] - 0s 30ms/step\n",
            "1/1 [==============================] - 0s 29ms/step\n",
            "1/1 [==============================] - 0s 29ms/step\n",
            "1/1 [==============================] - 0s 34ms/step\n",
            "1/1 [==============================] - 0s 29ms/step\n",
            "1/1 [==============================] - 0s 29ms/step\n",
            "1/1 [==============================] - 0s 30ms/step\n",
            "1/1 [==============================] - 0s 33ms/step\n",
            "1/1 [==============================] - 0s 30ms/step\n",
            "1/1 [==============================] - 0s 30ms/step\n",
            "1/1 [==============================] - 0s 33ms/step\n",
            "1/1 [==============================] - 0s 32ms/step\n",
            "1/1 [==============================] - 0s 31ms/step\n",
            "1/1 [==============================] - 0s 31ms/step\n",
            "1/1 [==============================] - 0s 29ms/step\n",
            "1/1 [==============================] - 0s 30ms/step\n",
            "1/1 [==============================] - 0s 30ms/step\n",
            "1/1 [==============================] - 0s 32ms/step\n",
            "1/1 [==============================] - 0s 32ms/step\n",
            "1/1 [==============================] - 0s 34ms/step\n",
            "1/1 [==============================] - 0s 31ms/step\n",
            "1/1 [==============================] - 0s 33ms/step\n",
            "1/1 [==============================] - 0s 31ms/step\n",
            "1/1 [==============================] - 0s 46ms/step\n",
            "1/1 [==============================] - 0s 32ms/step\n",
            "1/1 [==============================] - 0s 31ms/step\n",
            "1/1 [==============================] - 0s 35ms/step\n",
            "1/1 [==============================] - 0s 58ms/step\n",
            "1/1 [==============================] - 0s 36ms/step\n",
            "1/1 [==============================] - 0s 42ms/step\n",
            "1/1 [==============================] - 0s 64ms/step\n",
            "1/1 [==============================] - 0s 68ms/step\n",
            "1/1 [==============================] - 0s 63ms/step\n",
            "1/1 [==============================] - 0s 53ms/step\n",
            "1/1 [==============================] - 0s 74ms/step\n",
            "1/1 [==============================] - 0s 69ms/step\n",
            "1/1 [==============================] - 0s 57ms/step\n",
            "1/1 [==============================] - 0s 76ms/step\n",
            "1/1 [==============================] - 0s 58ms/step\n",
            "1/1 [==============================] - 0s 64ms/step\n",
            "1/1 [==============================] - 0s 64ms/step\n",
            "1/1 [==============================] - 0s 52ms/step\n",
            "1/1 [==============================] - 0s 61ms/step\n",
            "1/1 [==============================] - 0s 65ms/step\n",
            "If sailor tales to sailor tunes,pevn5wjv8lv;”1wx”gw!x .7”0!b(vo“;2u:8n’\n",
            "sz2-j!;z_(\n"
          ]
        }
      ],
      "source": [
        "print(extend_text(\"If sailor tales to sailor tunes,\", temperature=100))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "Shakespeare seems to be suffering from a heatwave. To generate more convincing text, a common technique is to sample only from the top k characters, or only from the smallest set of top characters whose total probability exceeds some threshold (this is called nucleus sampling ). Alternatively, you could try using beam search , which we will discuss later in this chapter, or using more GRU layers and more neurons per layer, training for longer, and adding some regularization if needed. Also note that the model is currently incapable of learning patterns longer than length , which is just 100 characters. You could try making this window larger, but it will also make training harder, and even LSTM and GRU cells cannot handle very long sequences. An alternative approach is to use a stateful RNN."
      ],
      "metadata": {
        "id": "AIK_qGI89-f9"
      }
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.6"
    },
    "nav_menu": {},
    "toc": {
      "navigate_menu": true,
      "number_sections": true,
      "sideBar": true,
      "threshold": 6,
      "toc_cell": false,
      "toc_section_display": "block",
      "toc_window_display": false
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
