# GuardRail Pro

GuardRail Pro is an enterprise-grade AI safety and evaluation framework designed to assess and monitor the safety, reliability, and ethical behavior of large language models (LLMs). It provides comprehensive testing for hallucination, bias, safety, and ethical reasoning, along with actionable recommendations for model improvement.

## Features

- **Hallucination Detection**: Evaluates factual accuracy using the TruthfulQA dataset.
- **Bias Detection**: Tests for gender and other biases using the WinoBias dataset.
- **Safety Testing**: Assesses toxicity and harmful content generation using the RealToxicityPrompts dataset.
- **Ethical Reasoning**: Evaluates ethical decision-making capabilities using the Hendrycks Ethics dataset.
- **Visual Reports**: Generates radar and bar charts for easy interpretation of test results.
- **Model Support**: Compatible with Hugging Face, OpenAI, and custom models.
- **Configurable Thresholds**: Customizable thresholds for each test category.
- **Historical Reports**: Saves and tracks audit results over time.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Het-Sathwara/guardrail-pro.git
   cd guardrail
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up configuration:
   - Edit `config/default.yaml` to customize thresholds and model settings.
   - Add your OpenAI API key (if using OpenAI models) in the interface.

## Usage

### Running the Interface
To launch the Gradio interface:
```bash
python guardrail_pro.py
```

### Key Components
1. **Model Audit**:
   - Load a model (Hugging Face, OpenAI, or custom).
   - Run a full audit to evaluate safety, bias, hallucination, and ethics.
   - View detailed results and recommendations.

2. **Configuration**:
   - Modify test thresholds and model parameters in the YAML configuration.

3. **Report History**:
   - View and compare historical audit reports.

### Supported Models
- **Hugging Face**: GPT-2, DialoGPT, GPT-Neo, etc.
- **OpenAI**: GPT-3.5, GPT-4, etc.
- **Custom**: Any locally hosted or custom-trained model.

## Example Workflow

1. Load a model (e.g., GPT-2 from Hugging Face).
2. Run a full audit.
3. Review the results:
   - **Recommendations**: Actionable steps to improve model safety.
4. Save the report for future reference.

## Configuration

Customize the following in `config/default.yaml`:
- Test sample size
- Thresholds for hallucination, bias, safety, and ethics
- Model parameters (temperature, max length, etc.)

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeatureName`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeatureName`).
5. Open a pull request.


## Support

For questions or issues, please [open an issue](https://github.com/Het-Sathwara/guardrail/issues) 

---

Thank you for using GuardRail Pro! 🚀
```
