# Drone Logo Mockup Tool

This is a web-based application that allows users to upload a logo and generate a drone light show mockup using 100 to 1000 drones.

## Features

- Upload PNG or JPG logo
- Choose drone count (100, 300, 500, 1000)
- Automatically extract outline
- Distribute drones along outline
- Render mockup and export as PNG

## Structure

```
drone_logo_mockup_repo/
├── backend/
│   ├── main.py               # FastAPI backend
│   └── requirements.txt      # Python dependencies
└── frontend/
    └── index.html            # HTML + JS UI
```

## How to Run Locally

### Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
Simply open `frontend/index.html` in your browser.

> Make sure the backend is running locally at http://localhost:8000

## Deployment Tips

- Host frontend on Vercel/Netlify
- Host backend on Render/Fly.io/Railway
- Adjust CORS as needed for public URLs

## Future Improvements

- Glow effects
- 3D drone animations
- Preset logo selection
- Show simulation timelines

---
Built with ❤️ for drone show innovators.