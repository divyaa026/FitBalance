# FitBalance AI-Powered Fitness Platform

Welcome to the FitBalance AI-powered fitness platform! This project aims to enhance your fitness journey by leveraging artificial intelligence to provide personalized workout plans and nutrition advice.

## Features

- **AI-generated workout plans:** Get custom workout routines tailored to your goals and fitness level.
- **Nutrition tracking:** Log meals, monitor nutrition, and receive AI-driven dietary recommendations.
- **Progress monitoring:** Track your workouts, nutrition, and progress over time.
- **Community support:** Join a community of like-minded fitness enthusiasts for motivation and accountability.

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- (Optional) Node.js and npm (if using the web front-end)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/divyaa026/FitBalance.git
   cd FitBalance
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install backend requirements:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the backend server:**
   ```bash
   cd backend/api
   python main.py
   ```

5. **(Optional) Run the front-end:**
   - If you have a front-end (e.g., React), follow its setup instructions in the `frontend/` directory.

## Usage

Once the backend is running, you can interact with the API at `http://localhost:5000/`.

To extend the platform, refer to the `docs/` directory for architecture, validation plans, and contributing guidelines.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): System architecture and overview.
- [`docs/clinical_validation_plan.md`](docs/clinical_validation_plan.md): Clinical validation plan.
- [`docs/user_study_protocol.md`](docs/user_study_protocol.md): User study protocol.
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md): How to contribute.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

For questions or support, open an issue or contact the maintainer at [divyaa026](https://github.com/divyaa026).