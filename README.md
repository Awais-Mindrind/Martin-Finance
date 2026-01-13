# 📘 **Martin Model Fine-Tuning System --- User Guide**

This guide explains how to update your AI model using new PDF or TXT
files.\
Everything runs automatically inside your RunPod GPU Pod using a
packaged Docker training system.

You do **not** need technical experience to use this.

------------------------------------------------------------------------

# 🧠 **What This System Does**

When you upload new documents and run the training command:

1.  ✔ Reads the PDFs/TXTs\
2.  ✔ Checks whether documents are usable\
3.  ✔ Turns each document into training data\
4.  ✔ Fine-tunes the Mistral 7B model using LoRA\
5.  ✔ Merges improvements into a new AI model\
6.  ✔ Saves the updated model to a **.gguf** file\
7.  ✔ Moves used PDFs into an archive folder\
    (so they will **not** be used again)

After training finishes, the CPU model you use for inference is
automatically updated.

------------------------------------------------------------------------

# 📂 **Where to Upload Training Documents**

Upload your documents to the following folder **inside your RunPod
Network Volume**:

    /workspace/data/raw_pdfs/

You can upload:

-   📄 PDF files\
-   ✍️ TXT files

> These files will be processed on the next training run.

### How to Upload Files in RunPod

1.  Go to your RunPod dashboard\
2.  Open **Volumes**\
3.  Open your GPU Pod's **Network Volume**\
4.  Navigate to:

```{=html}
<!-- -->
```
    /workspace/data/raw_pdfs/

5.  Click **Upload**\
6.  Select your PDF/TXT files\
7.  Done ✔

------------------------------------------------------------------------

# ⚙️ **Before Running Training: Environment Variables**

Your RunPod Pod must have these environment variables set:

### **MODEL_PATH**

Where the final updated model will be written.

Example:

    MODEL_PATH=/workspace/models/gguf/mistral.gguf

### **PDF_ARCHIVE_PATH**

Where used PDFs will be stored after training:

    PDF_ARCHIVE_PATH=/workspace/data/archive

### How to Set Them

In RunPod:

1.  Open your Pod\
2.  Click **Template Settings**\
3.  Scroll to **Environment Variables**\
4.  Add:

```{=html}
<!-- -->
```
    MODEL_PATH=/workspace/models/gguf/mistral.gguf
    PDF_ARCHIVE_PATH=/workspace/data/archive

5.  Save

These variables allow you to change model output locations **without
modifying code**.

------------------------------------------------------------------------

# 🚀 **How to Start a Full Training Run**

Once your PDFs are uploaded, run the following command inside your
RunPod container terminal:

``` bash
docker run --gpus all \
  -v /workspace:/workspace \
  awais2512/martin-model-tune \
  train_all
```

This command launches the full automated pipeline.

------------------------------------------------------------------------

# 🔄 **What Happens During train_all**

The system automatically performs these steps:

------------------------------------------------------------------------

### **1️⃣ Document Pre-Check**

-   Reads each PDF/TXT\

-   Checks if it is suitable for training\

-   Skips bad documents\

-   Saves results to

        /workspace/data/processed/pdf_pretest.json

------------------------------------------------------------------------

### **2️⃣ Dataset Creation**

-   Extracts text\

-   Splits text into smaller pieces\

-   Builds the final training dataset at

        /workspace/data/processed/train.jsonl

------------------------------------------------------------------------

### **3️⃣ LoRA Training (3 Levels)**

The system trains three separate improvements:

-   **Level 1** (small improvement)\
-   **Level 2** (medium)\
-   **Level 3** (strong)

Files saved under:

    /workspace/peft/level1/
     /workspace/peft/level2/
     /workspace/peft/level3/

------------------------------------------------------------------------

### **4️⃣ Evaluation**

The system tests:

-   Base model\
-   Level1\
-   Level2\
-   Level3

Results saved to:

    /workspace/eval/

------------------------------------------------------------------------

### **5️⃣ Merge Into Full Model**

The best layers are merged into the main Mistral HF model:

    /workspace/peft/merged/

------------------------------------------------------------------------

### **6️⃣ Convert to GGUF**

The final merged model is converted into `.gguf` format and written to:

    $MODEL_PATH

Example:

    /workspace/models/gguf/mistral.gguf

This is the file used by your CPU inference system.

------------------------------------------------------------------------

# ✅ **Test GGUF (Base + Adapters)**

Use these commands inside the GPU pod or CPU pod terminal to test the
base model and adapters with the `test_gguf` runner:

### Base model only

``` bash
python /app/main.py test_gguf \
  --model /workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

### Single adapter (example: B2.gguf)

``` bash
python /app/main.py test_gguf \
  --model /workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --adapter /workspace/output/adapters_gguf/v3/B2.gguf
```

### Test all adapters in a folder

``` bash
python /app/main.py test_gguf \
  --model /workspace/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf \
  --adapters-dir /workspace/output/adapters_gguf/v3
```

Optional flags:
- `--prompt "your prompt here"`
- `--max-tokens 256`
- `--temp 0.7`
- `--ngl 0` (force CPU if no GPU)

------------------------------------------------------------------------

### **7️⃣ Archive Processed PDFs**

Your input documents are moved to:

    /workspace/data/archive/

This prevents training the same document more than once.

New training runs only use **newly uploaded files**.

------------------------------------------------------------------------

# 🧾 **Summary of Folders**

  -----------------------------------------------------------------------
  Folder                            Purpose
  --------------------------------- -------------------------------------
  `/workspace/data/raw_pdfs/`       Upload new PDFs/TXTs here

  `/workspace/data/archive/`        Processed PDFs are moved here
                                    automatically

  `/workspace/data/processed/`      Training dataset + pretest logs

  `/workspace/peft/level*/`         LoRA adapters (Levels 1--3)

  `/workspace/peft/merged/`         Merged HF model before GGUF

  `/workspace/models/gguf/`         Final `.gguf` model

  `/workspace/eval/`                Evaluation outputs
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 🔁 **Running Training Again**

When you have more training documents:

1.  Upload new PDFs/TXTs to

        /workspace/data/raw_pdfs/

2.  Run the same command again:

``` bash
docker run --gpus all \
  -v /workspace:/workspace \
  awais2512/martin-model-tune \
  train_all
```

Everything else happens automatically.

------------------------------------------------------------------------

# ❓ **Frequently Asked Questions**

### **Do I need to remove old files?**

No. They are moved to `/workspace/data/archive/` automatically.

------------------------------------------------------------------------

### **Does the model overwrite the old version?**

Yes --- the file at `$MODEL_PATH` is replaced with the new fine-tuned
model.

------------------------------------------------------------------------

### **Do I need to edit the Dockerfile?**

No --- all paths are dynamic using environment variables.

------------------------------------------------------------------------

### **Can I version models?**

Yes. Just change the MODEL_PATH variable in RunPod:

    MODEL_PATH=/workspace/models/gguf/mistral-v2.gguf

------------------------------------------------------------------------

# 🎉 **You're Ready to Fine-Tune Your Own AI Model**

The system is designed to:

-   Be simple\
-   Require no coding\
-   Update itself\
-   Avoid re-training old documents\
-   Produce a ready-to-use `.gguf` model

If you need help customizing or extending the system (web UI, scheduled
training, automatic deployment), we can do that too.

------------------------------------------------------------------------
