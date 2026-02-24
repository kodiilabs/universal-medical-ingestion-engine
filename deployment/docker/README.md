# Medical Document Analyzer

## What This Does

Upload medical documents (lab reports, prescriptions, radiology reports) and the app will automatically extract, classify, and structure all the data. You can also chat with the AI about your documents.

---

## Setup (One Time)

### Step 1: Install Docker Desktop

1. Go to **https://www.docker.com/products/docker-desktop/**
2. Click **"Download for Windows"**
3. Run the installer and follow the prompts
4. **Restart your computer** when asked

### Step 2: Start the App

1. Open the folder you received
2. Go into the **`deployment\docker`** folder
3. **Double-click `start.bat`**

The first time takes **10-15 minutes** (it downloads AI models). You'll see progress in the window. When it's done, your browser will open automatically.

---

## Daily Use

| To do this... | Do this... |
|---|---|
| **Start the app** | Double-click `deployment\docker\start.bat` |
| **Stop the app** | Double-click `deployment\docker\stop.bat` |

After the first time, starting takes about **15 seconds**.

---

## Using the App

1. Open your browser to **http://localhost:3000** (opens automatically)
2. Click **"Upload"** and select a medical document (PDF, JPG, PNG)
3. Wait for processing (30 seconds to 2 minutes)
4. View the extracted data in the tabs:
   - **Extracted Values** -- all test results, medications, etc.
   - **Clinical Summary** -- AI-generated summary
   - **FHIR** -- standardized medical data format
   - **Chat** -- ask questions about the document

Sample documents are included so you can test right away using the **Samples** page.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Docker is not installed" | Install Docker Desktop (see Step 1 above) |
| "Docker did not start in time" | Open Docker Desktop manually, wait for it to finish loading, then run start.bat again |
| App is slow on first document | Normal -- AI models load into memory on first use. Second document will be faster. |
| Browser shows blank page | Wait 30 seconds and refresh. The frontend may still be starting. |
| "Cannot connect to backend" | Make sure start.bat finished without errors. Try stop.bat then start.bat again. |

---

## System Requirements

- **Windows 10/11** (64-bit)
- **16 GB RAM** (minimum)
- **15 GB free disk space** (for AI models)
- **Internet connection** (first time only, for Docker + model downloads)
