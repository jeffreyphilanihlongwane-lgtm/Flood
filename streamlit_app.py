import streamlit as st
from audiocraft.models import MusicGen

st.set_page_config(page_title="FLOOD AI", page_icon="🎧")
st.title("🔥 FLOOD AI - SUNO QUALITY")
st.write("Generates 2 beats like Suno - HQ studio")

@st.cache_resource
def load_model():
    m = MusicGen.get_pretrained('facebook/musicgen-stereo-small')
    m.set_generation_params(duration=30, use_sampling=True, top_k=250, top_p=0.9, cfg_coef=6)
    return m

model = load_model()

prompt = st.text_area("Prompt", "High quality studio mix, Zulu Afro-Trap anthem, soulful piano, heavy 808, log drum, 92 BPM, professional mastering, wide stereo", height=120)

if st.button("🚀 GENERATE 2 BEATS"):
    with st.spinner("Cooking 2 hits... 1 min"):
        wavs = model.generate([prompt, prompt])
        for i in range(2):
            st.subheader(f"Beat {i+1}")
            st.audio(wavs[i][0].cpu().numpy().T, sample_rate=model.sample_rate)
