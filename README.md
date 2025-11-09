# bgremover

Streamlit app to remove backgrounds and resize images (PNG output).

## Deploy notes

- The app's background-removal feature depends on `rembg` and `onnxruntime`, which are heavy native packages and may fail to install on Streamlit Cloud without additional system packages.
- For a safe deploy that ensures the UI and upload/download flows work, `rembg` and `onnxruntime` are left commented out in `requirements.txt` on the `deploy-ready` branch.
- If you want the background removal in production, re-enable these lines in `requirements.txt` and ensure `packages.txt` includes system packages (we've suggested some common ones: `libgomp1`, `libopenblas-dev`, `cmake`, `build-essential`). If you still get build errors, paste the Deployment logs here and we'll iterate.

## Quick deploy steps

1. In Streamlit Cloud, create a new app and select the repo `Talpedreros/bgremover`.
2. Choose branch `deploy-ready` and `app.py` as the main file. Deploy.
3. If deploy fails, open **Logs** → **Deployment logs** and copy the error block here.

If you want me to re-enable `rembg`/`onnxruntime` and attempt to add system deps, give me the OK and I'll prepare a branch with those changes instead.
