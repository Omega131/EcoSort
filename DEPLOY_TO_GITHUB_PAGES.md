# 🌐 1-Click Deployment to GitHub Pages (100% Free Forever)

Deploy **EcoSort** online for free using **GitHub Pages**! 

Because the model is only **1.31 MB**, it runs **100% client-side in the user's browser using WebAssembly (`onnxruntime-web`)**.
- ✅ **100% Free Forever** (No credit cards, no billing, no cloud costs)
- ✅ **Zero Server Management** (No server sleep, no cold starts, high global availability via GitHub CDN)
- ✅ **100% Privacy-Preserving** (Photos & camera frames never leave the user's phone or computer)
- ✅ **Instant Real-Time Inference** (Directly on device CPU / GPU via WASM)

---

## 🚀 2-Minute Setup Steps

### Step 1: Push the `docs/` folder to GitHub
In your local terminal inside `EcoSort`, run:

```bash
git add docs/ README.md DEPLOY_TO_GITHUB_PAGES.md
git commit -m "Add GitHub Pages in-browser AI web app"
git push origin main
```

---

### Step 2: Enable GitHub Pages in your Repository Settings
1. Open your repository on GitHub (`https://github.com/<YOUR_USERNAME>/EcoSort`).
2. Click on **Settings** (top tab).
3. In the left sidebar, click on **Pages** (under the *Code and automation* section).
4. Under **Build and deployment**:
   - **Source:** Select **`Deploy from a branch`**.
   - **Branch:** Select **`main`** from the branch dropdown.
   - **Folder:** Change `/ (root)` to **`/docs`** from the folder dropdown.
5. Click **Save**.

---

### Step 3: Your App is Live Online!
GitHub will deploy the app in under 60 seconds. Your live public URL will be:

👉 **`https://<YOUR_USERNAME>.github.io/EcoSort/`**

Anyone on any smartphone, tablet, or laptop can now open this link and:
- Upload waste images
- Use their live camera / webcam
- Test sample waste items
- Get instant category predictions and actionable eco recycling guidance!
