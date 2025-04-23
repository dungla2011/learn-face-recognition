# Facial Recognition Project

## Overview
This project implements a facial recognition application that processes images to identify individuals and track their IDs. It is designed to manage both new and known individuals, storing their images and associated ID information.

## Project Structure
```
facial-recognition-project
├── main.py               # Main entry point for the application
├── saved_images          # Directory for storing images
│   ├── known            # Images of known individuals
│   └── new              # Images of new individuals
├── id_tracker.json       # JSON file for tracking IDs
└── README.md             # Project documentation
```

## Setup Instructions
1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies (if any) using pip:
   ```
   pip install -r requirements.txt
   ```

## Usage Guidelines
- Place images of new individuals in the `saved_images/new` folder.
- Images of known individuals should be stored in the `saved_images/known` folder.
- The `id_tracker.json` file will automatically update with the ID information as the application processes images.
- Run the application using:
   ```
   python main.py
   ```

## Contributing
Feel free to submit issues or pull requests to improve the project.