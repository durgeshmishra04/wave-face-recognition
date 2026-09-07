# ALCON Surveillance Backend

Start the application with:

```powershell
python app.py
```

Camera selection is configuration-driven. Set `ENABLED_CAMERAS` in `.env`:

```env
ENABLED_CAMERAS=CAM001
ENABLED_CAMERAS=CAM001,CAM002,CAM003
ENABLED_CAMERAS=ALL
```

Camera definitions are stored in `config/cameras.json`. The existing AI,
tracking, event, database, notification, API, and Socket.IO implementations
remain behaviorally compatible while the package structure is introduced in
stages.
