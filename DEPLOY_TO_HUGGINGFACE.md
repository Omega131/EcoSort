# 🚀 Deploying EcoSort Online with Hugging Face Spaces (100% Free)

This guide walks you through deploying **EcoSort** to **Hugging Face Spaces** so anyone in the world can access your web application on their phone or laptop with a public HTTPS link (e.g. `https://<your-username>-ecosort.hf.space`).

---

## ⚡ Method 1: Git Push directly to Hugging Face (Recommended — 2 Minutes)

### Step 1: Create a Free Hugging Face Space
1. Sign up or log in at [Hugging Face](https://huggingface.co/).
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
3. Fill in the details:
   - **Space Name:** `EcoSort` (or any name you like)
   - **License:** `mit`
   - **Select the Space SDK:** Choose **Docker** &rarr; **Blank**
   - **Space Hardware:** Free (CPU basic &bull; 2 vCPU &bull; 16 GB RAM)
   - **Visibility:** **Public**
4. Click **Create Space**.

---

### Step 2: Push your Code to the Space
In your local terminal inside `EcoSort`, run:

```bash
# 1. Add your Hugging Face space as a git remote (replace <YOUR_USERNAME> with your HF username):
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/EcoSort

# 2. Push to Hugging Face:
git push --force space main
```

*(If prompted for credentials, use your Hugging Face username and your Access Token as password from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).*

---

### Step 3: That's It!
Hugging Face will detect the [`Dockerfile`](Dockerfile) and automatically:
1. Build the lightweight container image.
2. Load `model.onnx` into memory.
3. Start the FastAPI server on port 7860.
4. Give you a permanent public URL: `https://<YOUR_USERNAME>-ecosort.hf.space`!

---

## 🌐 Method 2: Link with GitHub (Continuous Deployment)

If your EcoSort repository is already on GitHub:

1. Create a new Space on [huggingface.co/new-space](https://huggingface.co/new-space) (choose **Docker** &rarr; **Blank**).
2. Go to **Settings** of your Space.
3. Under **Repository mirroring** or **GitHub Actions**:
   - Set up automatic sync with your GitHub repository.
   - Every time you run `git push origin main`, your public web app will automatically rebuild and update!

---

## ⚡ Bonus: Want an Instant Public Link RIGHT NOW? (No Sign-Up)

If you want an immediate public HTTPS link to test on your phone or send to a colleague without setting up accounts:

Make sure the local server is running (`python run_app.py`), then run in a separate terminal:

```bash
npx -y localtunnel --port 8000
```

You'll instantly receive a public URL like:
`https://ecosort-xxxx.loca.lt`
Open it on any mobile device or computer to start classifying waste live!
