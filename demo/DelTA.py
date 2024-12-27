import streamlit as st
import subprocess
import os
import re
# from utils import get_unique_filename, extract_progress_info

def get_unique_filename(directory, filename):
    """
    If the file exists, append an incremental suffix (e.g., _1, _2) to the filename until a unique name is found.
    """
    base_name, extension = os.path.splitext(filename)  # Separate filename and extension
    counter = 1
    unique_name = filename
    while os.path.exists(os.path.join(directory, unique_name)):
        unique_name = f"{base_name}_{counter}{extension}"
        counter += 1
    return unique_name

def extract_progress_info(line):
    """
    Extract the percentage and remaining time from the progress information.
    Example input: "53%|███████████████████████████| 29/55 [00:55<00:38, 1.47s/it]"
    Return value: progress: 53, remaining_time: "00:38"
    """
    progress = None
    remaining_time = None

    # Use regular expressions to extract the percentage and remaining time
    match = re.search(r"(\d+)%.*<([^,>]+)", line)
    if match:
        progress = int(match.group(1))  # Extract the percentage and convert to an integer
        remaining_time = match.group(2)  # Extract the remaining time
    return progress, remaining_time

def run_shell_script_with_progress(lang, use_model, src, file_name):
    """
    Call the Shell script and update progress information in real-time.
    """
    shell_script = "./run_gpt.sh" if use_model.startswith("gpt") else "./run_qwen.sh"
    cmd = [shell_script, lang, use_model, src]

    # File processing status
    st.write(f"### 📄 Processing: `{file_name}`")

    # Placeholder
    progress_bar = st.progress(0)
    progress_status = st.empty()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        output_value = None

        for line in iter(process.stderr.readline, ''):
            line = line.strip()
            # Extract progress information
            if "%|" in line and "checkpoint" not in line and "Starting" not in line:
                progress, remaining_time = extract_progress_info(line)
                if progress is not None and remaining_time is not None:
                    progress_bar.progress(progress / 100)
                    progress_status.write(f"🕒 {progress}% | Estimated Time Left: {remaining_time}")

        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line.startswith("OUTPUT="):
                output_value = line[len("OUTPUT="):]

        process.wait()

        if process.returncode != 0:
            st.error("❌ Translation failed. Please check your input.")
            return None

        # Clear the progress bar and status
        progress_bar.empty()
        progress_status.empty()
        return output_value

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
        return None

# Sidebar configuration
st.sidebar.title("💬 DelTA")
st.sidebar.markdown("**Document-levEL Translation Agent**")
lang = st.sidebar.selectbox("🌐 Language Pair:", 
    ["de-en", "en-de", "en-fr", "en-jr", "en-zh", "fr-en", "ja-en", "zh-en"], index=4)
use_model = st.sidebar.selectbox("🤖 LLM Model:", 
    ["Qwen2-7B-Instruct", "Qwen2-72B-Instruct", "gpt-3.5-turbo", "gpt-4o-mini"], index=0)
st.sidebar.markdown(
        """
        **Additional Resources**  
        [📄 Our Paper](https://arxiv.org/abs/2410.08143)  
        [💻 Source Code](https://github.com/YutongWang1216/DocMTAgent)
        """
    )
# Main content area
st.title("💡 AI Document Translator")
st.markdown("### Translate text or documents with powerful AI models!")

# File upload area
uploaded_files = st.file_uploader("📂 Upload your TXT, WORD, or PDF files:", 
    accept_multiple_files=True, type=["txt", "pdf", "docx"])
text_input = st.text_area("📝 Or enter text manually:", 
    placeholder="Type or paste your text here...", height=68)

if not lang or not use_model or (not uploaded_files and not text_input):
    st.warning("⚠️ Please select options and provide input before starting.")
    st.stop()

# Start button
if st.button("🚀 Start Translation"):
    # File processing and translation
    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_dir = "data/input/"
            unique_filename = get_unique_filename(save_dir, uploaded_file.name)
            save_path = os.path.join(save_dir, unique_filename)

            # Save the file
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())

            # Translate
            output = run_shell_script_with_progress(lang, use_model, save_path, uploaded_file.name)
            if output:
                # Use the filename as a unique key
                unique_key = f"text_area_{uploaded_file.name}"
                with st.expander(f"📃 Translated Content: {uploaded_file.name}"):
                    with open(output, "r", encoding="utf-8") as file:
                        st.text_area(
                            "Translated Text:", 
                            value=file.read(), 
                            height=300, 
                            key=unique_key
                    )        
    # Text input translation
    if text_input:
        save_dir = "data/input/"
        save_path = os.path.join(save_dir, "user_input.txt")
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text_input)

        output = run_shell_script_with_progress(lang, use_model, save_path, "User Input")
        if output:
            # Assign a unique key for user input
            with st.expander("📃 Translated Content: User Input"):
                with open(output, "r", encoding="utf-8") as file:
                    st.text_area(
                        "Translated Text:", 
                        value=file.read(), 
                        height=300, 
                        key="user_input_text_area"
                    )