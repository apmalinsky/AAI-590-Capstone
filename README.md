# AAI-590-Capstone
TODO: Project description



# ⬇️ Downloading Your Kaggle API Key

To access competition and dataset files programmatically, you must first download your personal **Kaggle API key**.

---

## Steps to Get Your `kaggle.json` File

Follow these steps to generate and download your key:

1.  **Log in** to your Kaggle account on the website.
2.  Navigate to your **Settings** page (Click your profile picture in the top-right corner, then select **Settings**).
3.  Scroll down the page until you find the **API** section.
4.  Click the **"Create New Token"** button.

> 💡 **Result:** This action automatically downloads a file named **`kaggle.json`** to your computer. This file contains your credentials and is essential for authentication.


# 🔑 Setting Up Your Private Kaggle API Credentials

This guide shows you how to securely store your `kaggle.json` API key as a **private dataset** on Kaggle, which can then be easily accessed in your notebooks.

---

## 1. Create Your Private Credentials Dataset (One-Time Setup)

You only have to perform this step once.

* On the Kaggle website, navigate to the **Datasets** section (via the left-hand menu).
* Click **"New Dataset"**.
* Give it a simple, memorable title, such as **`my-kaggle-secret`**.
* Drag-and-drop your **`kaggle.json`** file into the uploader area.
* **Crucially**, ensure you mark the dataset as **Private**.
* Click **"Create"**.

> 💡 **Result:** You now have a private dataset that securely holds your API key.

---

## 2. Add Your Secret Dataset to a Notebook

* In your Kaggle notebook, click the **"+ Add Data"** button in the top-right corner.
* Go to the **"Your Datasets"** tab.
* Find your dataset (e.g., **`my-kaggle-secret`**) and click **"Add"**.

> **Access Path:** The API key will now be available in your notebook at a path similar to:
>
> `/kaggle/input/my-kaggle-secret/`
