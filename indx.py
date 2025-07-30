import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import os

@st.cache_data
def load_glove_embeddings(glove_file_path="glove.6B.50d.txt"):
    embeddings = {}
    with open(glove_file_path, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

def embed_job_title(title, embeddings, embedding_dim=50):
    words = title.lower().split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if not vectors:
        return np.zeros(embedding_dim)
    return np.mean(vectors, axis=0)

def is_valid_email(email: str) -> bool:
    regex = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
    return re.match(regex, email) is not None

class LeadScoringModel(nn.Module):
    def __init__(self, input_dim=52):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x.squeeze()

@st.cache_resource
def load_model():
    model = LeadScoringModel()
    if os.path.exists("lead_scoring_model.pth"):
        model.load_state_dict(torch.load("lead_scoring_model.pth", map_location=torch.device("cpu")))
        st.success("Loaded trained model from 'lead_scoring_model.pth'")
    else:
        st.warning("No saved model found — using random weights")
    return model

def score_lead_ai(lead: dict, model, embeddings) -> float:
    title_emb = embed_job_title(lead.get('job_title', ''), embeddings)
    company_size = lead.get('company_size', 0) / 500
    email_valid = 1 if is_valid_email(lead.get('email', '')) else 0
    features = np.concatenate([title_emb, [company_size, email_valid]])
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        score = model(features_tensor).item()
    return score

def deduplicate_leads_df(df):
    return df.drop_duplicates(subset=['email'])

def main():
    st.title("AI Lead Scoring Tool")

    st.markdown("""
Upload a CSV file with the following columns:
- `name`
- `email`
- `job_title`
- `company`
- `company_size`
- `domain`
    """)

    uploaded_file = st.file_uploader("Upload Leads CSV", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"{len(df)} rows loaded.")

        required_columns = {'email', 'job_title', 'company_size'}
        if not required_columns.issubset(df.columns):
            st.error("CSV must include: 'email', 'job_title', 'company_size'")
            return

        df = deduplicate_leads_df(df)
        st.info(f"{len(df)} unique leads after deduplication")

        model = load_model()
        embeddings = load_glove_embeddings()

        scores = []
        for _, row in df.iterrows():
            lead = row.to_dict()
            score = score_lead_ai(lead, model, embeddings)
            scores.append(score)

        df['score'] = scores
        df = df.sort_values(by='score', ascending=False)

        st.subheader("Scored Leads")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Scored Leads", csv, "scored_leads.csv", "text/csv")

if __name__ == "__main__":
    main()
