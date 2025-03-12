# AlphaAnon

AlphaAnon is a project designed to fine-tune a pre-trained language model on a sample 4chan dataset before applying it to live data using GRPO.

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- PyTorch
- Transformers
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/AlphaAnon.git
    cd AlphaAnon
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Fine-Tuning the Model

Run the fine-tuning script:
    ```sh
    python fine_tune.py
    ```

### Using GRPO on Live Data

1. Run the 4chan_data script to fetch live data, which will output 4chan_thread_dataset.json
2. Change the dataset source inside grpo.py to "4chan_thread_dataset.json"
3. Run the GRPO script:
    ```sh
    python grpo.py
    ```