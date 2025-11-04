import json
import os
import time
import streamlit as st
import torch
import pandas as pd

from models import MLPNextWord, RNNNextWord, GRUNextWord, LSTMNextWord
from infer import generate_words, get_top_k_probs, sample_next

# Get the directory where app.py is located
APP_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Next-word Visualizer",
                   page_icon="🧠", layout="wide")
st.title("Next-word Visualizer")

# ----------------------------
# SIDEBAR - DECODING SETTINGS
# ----------------------------
with st.sidebar:
    st.header("📋 Model Selection")
    dataset_choice = st.selectbox(
        "Dataset/Model",
        options=["Sherlock", "C++"],
        help="Select the trained model checkpoint to use"
    )

    checkpoint_dir = "checkpoints_sherlock" if dataset_choice == "Sherlock" else "checkpoints_cpp"
    checkpoint_dir = os.path.join(
        APP_DIR, checkpoint_dir)  # Make path absolute
    ckpt_path = os.path.join(checkpoint_dir, "best.pt")
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")

    model_type = st.selectbox(
        "Model Architecture",
        options=["MLP", "RNN", "GRU", "LSTM"],
        index=0,
        help="Select the model architecture"
    )

    st.divider()

    st.header("⚙️ Decoding Settings")

    strategy = st.selectbox(
        "Decoding strategy",
        options=["greedy", "sampling", "top_k", "top_p"],
        index=0,
        help="Greedy: always picks highest probability\n"
             "Sampling: samples from full distribution\n"
             "Top-k: samples from top k tokens\n"
             "Top-p: nucleus sampling"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Higher temperature = more randomness"
    )

    top_k = st.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        disabled=(strategy not in ["top_k", "sampling"]),
        help="Sample from top k tokens only"
    )

    top_p = st.slider(
        "Top-p (nucleus)",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        disabled=(strategy != "top_p"),
        help="Nucleus sampling: cumulative probability threshold"
    )

    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=1,
        max_value=100,
        value=20,
        step=1,
        help="Maximum number of tokens to generate"
    )

    seed = st.number_input(
        "Random seed (0 = random)",
        min_value=0,
        value=0,
        help="Set seed for reproducibility"
    )

    st.divider()
    checkpoint_dir_display = "checkpoints_sherlock" if dataset_choice == "Sherlock" else "checkpoints_cpp"
    st.caption(f"📁 Checkpoint: {checkpoint_dir_display}")
    if not os.path.exists(ckpt_path):
        st.error(f"⚠️ Checkpoint not found: {ckpt_path}")
    if not os.path.exists(vocab_path):
        st.error(f"⚠️ Vocab not found: {vocab_path}")


# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_model_and_vocab(checkpoint_dir, model_type):
    """Load model and vocabulary from checkpoint directory."""
    ckpt_path = os.path.join(checkpoint_dir, "best.pt")
    vocab_path = os.path.join(checkpoint_dir, "vocab.json")

    if not os.path.exists(ckpt_path) or not os.path.exists(vocab_path):
        return None, None, None

    # Load vocabulary
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    stoi = vocab_data["word2idx"]
    itos = {int(k): v for k, v in vocab_data["idx2word"].items()}
    context_len = vocab_data.get("context_len", 5)

    # Load model checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Select model architecture
    model_classes = {
        "MLP": MLPNextWord,
        "RNN": RNNNextWord,
        "GRU": GRUNextWord,
        "LSTM": LSTMNextWord,
    }

    model_class = model_classes.get(model_type, MLPNextWord)

    # Determine model config based on checkpoint directory name
    checkpoint_dir_name = os.path.basename(checkpoint_dir)

    # Create model (you may need to adjust these based on your training config)
    if checkpoint_dir_name == "checkpoints_sherlock":
        model = model_class(
            vocab_size=len(stoi),
            block_size=context_len,
            emb_dim=64,
            hidden_size=1024,
            num_hidden_layers=4,
            activation="relu",
            dropout=0.0,
        ).to(device)
    elif checkpoint_dir_name == "checkpoints_cpp":
        model = model_class(
            vocab_size=len(stoi),
            block_size=context_len,
            emb_dim=64,
            hidden_size=512,
            num_hidden_layers=4,
            activation="relu",
            dropout=0.0,
        ).to(device)

    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    return model, stoi, itos


# ----------------------------
# MAIN CONTENT
# ----------------------------
st.subheader("Prompt")
prompt_str = st.text_input(
    "Enter context words (space-separated)",
    value="the quick brown",
    help="Enter words separated by spaces"
)
prompt_tokens = [t for t in prompt_str.strip().lower().split() if t]

if not prompt_tokens:
    st.warning("Please enter a prompt")
    st.stop()

# Load model
model, stoi, itos = load_model_and_vocab(checkpoint_dir, model_type)

if model is None:
    checkpoint_dir_name = os.path.basename(checkpoint_dir)
    st.error(
        f"Failed to load model from {checkpoint_dir_name}. Make sure checkpoints exist.")
    st.info(f"Expected paths:\n- {ckpt_path}\n- {vocab_path}")
    st.stop()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# TOP-10 NEXT TOKEN PROBABILITIES
# ----------------------------
st.subheader("Top-10 Next Token Probabilities")

if st.button("Show next-token probabilities"):
    model.eval()
    unk = stoi.get("<unk>", 1)
    pad = stoi.get("<pad>", 0)
    block_size = model.block_size

    # Prepare context
    context = [pad] * max(0, block_size - len(prompt_tokens)) + [
        stoi.get(w, unk) for w in prompt_tokens
    ][-block_size:]

    with torch.no_grad():
        x = torch.tensor(context, dtype=torch.long, device=device).view(1, -1)
        logits = model(x)[0]
        top_probs = get_top_k_probs(logits, k=10, temperature=temperature)

    # Display token IDs for prompt
    prompt_ids = [stoi.get(w, unk) for w in prompt_tokens]
    st.write("**Input tokens:**")
    token_display = " | ".join(
        [f"{w} ({stoi.get(w, unk)})" for w in prompt_tokens])
    st.write(token_display)

    # Display top probabilities
    st.write("**Top-10 next token probabilities:**")
    prob_data = []
    for idx, prob in top_probs:
        token = itos.get(idx, "<unk>")
        prob_data.append({"Token": token, "ID": idx,
                         "Probability": f"{prob:.4f}"})

    df = pd.DataFrame(prob_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ----------------------------
# STEP-BY-STEP GENERATION
# ----------------------------
st.divider()
st.subheader("Step-by-Step Generation")

# Map strategy
strategy_map = {
    "greedy": "greedy",
    "sampling": "sampling",
    "top_k": "top_k",
    "top_p": "top_p",
}
decoding_strategy = strategy_map.get(strategy, "greedy")

if st.button("Generate step-by-step"):
    st.write(f"**Decoding:** {strategy.capitalize()}")

    model.eval()
    unk = stoi.get("<unk>", 1)
    pad = stoi.get("<pad>", 0)
    block_size = model.block_size

    # Prepare initial context
    context = [pad] * max(0, block_size - len(prompt_tokens)) + [
        stoi.get(w, unk) for w in prompt_tokens
    ][-block_size:]

    generated = []
    g = None
    if seed > 0:
        g = torch.Generator(device=device)
        g.manual_seed(seed)

    # Container for streaming output
    output_container = st.empty()
    current_text = " ".join(prompt_tokens)

    with torch.no_grad():
        for step in range(max_new_tokens):
            x = torch.tensor(context, dtype=torch.long,
                             device=device).view(1, -1)
            logits = model(x)[0]

            # Get next token
            if strategy == "sampling":
                actual_strategy = "sampling"
            elif strategy == "top_k":
                actual_strategy = "top_k"
            elif strategy == "top_p":
                actual_strategy = "top_p"
            else:
                actual_strategy = "greedy"

            ix = sample_next(
                logits,
                strategy=actual_strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generator=g,
            )

            token = itos.get(ix, "<unk>")

            # Termination logic
            if token == "." or token == "<pad>" or token == "<unk>":
                break

            generated.append(token)
            current_text = " ".join(prompt_tokens + generated)

            # Update display
            output_container.write(f"**Generated:** {current_text}")

            # Small delay for visualization
            time.sleep(0.1)

            # Update context
            context = context[1:] + [ix]

    # Final output
    st.success(f"**Final output:** {current_text}")

    if generated:
        st.write(
            f"**Generated {len(generated)} tokens:** {', '.join(generated)}")
