# AlphaAnon

AlphaAnon is a project designed to fine-tune a pre-trained language model on a sample 4chan dataset before applying it to live data using GRPO.
Default model and settings tested on consumer hardware, fits in 8GB VRAM. Feed your local model nice threads everyday like a tomagachi for best results :3

## Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- PyTorch
- Transformers
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
    ```
    git clone https://github.com/yourusername/AlphaAnon.git
    cd AlphaAnon
    ```

2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

3. Fine-Tuning the Model

Get 4chan data
``` python 4chan_data.py ```

Run the fine-tuning script:
    ``` python fine_tune.py ```

3. Run the GRPO script:
    ```python grpo.py ```
