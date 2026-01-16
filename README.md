# AccessibleVision

A bilingual, high-contrast web platform that converts visual input into audio descriptions for visually impaired users.

## Features
- **Scenery Mode**: Analyze static images
- **Walking Mode**: Process video for navigation guidance
- **Bilingual**: English and Hindi support
- **Accessible**: WCAG 2.1 AA compliant
- **Cloud-Native**: Deploys on Streamlit Cloud

## Quick Start

### Local Testing
```bash
pip install -r requirements.txt
streamlit run app.py
```

Visit http://localhost:8501

### Deployment
1. Push to GitHub
2. Go to streamlit.io/cloud
3. Select repository
4. Deploy

## Configuration

Update `MODEL_CONFIG` in `app.py` with your Google Drive model URLs.

## License
MIT License
