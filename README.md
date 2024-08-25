# EfficientNet B2 Image Classification

This project implements an image classification model using the EfficientNet B2 architecture, fine-tuned on a custom dataset. It provides a modular and easy-to-use structure for training and evaluating the model.

## Project Structure

```
project_root/
│
├── data/
│   ├── train/
│   └── test/
│
├── src/
│   ├── __init__.py
│   ├── data_setup.py
│   ├── train_and_test.py
│   ├── model.py
│
├── main.py
├── requirements.txt
└── README.md
```

- `data/`: Contains the training and testing datasets.
- `src/`: Source code for the project.
- `main.py`: The entry point of the project.

## Setup

1. Clone the repository:

   ```
   git clone https://github.com/brepositorium/effnetb2-sentiment-analysis.git
   cd effnetb2-sentiment-analysis
   ```

2. Create a virtual environment and activate it:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the model, run:

```
python main.py
```

This will start the training process using the EfficientNet B2 model on your dataset. The script will output training progress and final results.

## Customization

- Edit `src/model.py` to experiment with different model architectures or layer configurations.
- Adjust data augmentation in `src/data_setup.py` if needed.

## Results

After training, the model will output training and validation accuracy and loss. You can find these results printed in the console output.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements or encounter any problems.

## License

MIT License
