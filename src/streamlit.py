"""Streamlit UI: streamlit run src/streamlit.py"""

import streamlit as st

from predict import predict

st.markdown("""
<style>
  input[type=text] { border-color: #4a90d9 !important; }
  input[type=text]:focus { border-color: #1a6fc4 !important; box-shadow: 0 0 0 2px #4a90d933 !important; }
</style>
""", unsafe_allow_html=True)

st.title("Phishing URL Detector")

url = st.text_input("Enter a URL to check")

if st.button("Check") and url:
    result = predict(url)
    is_phishing = result["label"] == "Phishing"
    color = "red" if is_phishing else "green"
    icon  = "🚨" if is_phishing else "✅"

    st.markdown(
        f"<div style='padding:1rem;border-radius:8px;background:{color}22;border:2px solid {color}'>"
        f"<h2 style='color:{color};margin:0'>{icon} {result['label']}</h2>"
        f"<p style='margin:0.25rem 0 0'>Confidence: <strong>{result['confidence']:.1%}</strong></p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)

    with st.expander("Feature breakdown"):
        st.dataframe(
            {"Feature": list(result["features"].keys()),
             "Value":   list(result["features"].values())},
            use_container_width=True,
        )
