## AI-Powered Lead Scoring Tool

## Overview
This project implements an AI-powered lead scoring tool designed to help sales teams prioritize high-quality leads. It uses a simple neural network trained on synthetic data with features derived from job titles, company size, and email validity. The model assigns a lead quality score between 0 and 1 to help optimize outreach efforts.
A user-friendly Streamlit web app allows users to upload CSV files of leads, deduplicates entries by email, scores each lead with the AI model, and lets users download the scored list for sales workflows.

## Features
1. AI lead quality scoring using a PyTorch MLP model
2. Input features:
  GloVe 50-dimensional embeddings of job titles
  Normalized company size
  Email validity check with regex
3. Deduplication of leads by email address
4. Simple Streamlit interface for uploading CSV and downloading scored leads
5. Lightweight and fast, suitable for real-time scoring

## Installation
1.	Clone the repository
2.	Install required packages:
pip install -r requirements.txt
3.	Download GloVe embeddings (50d) from https://nlp.stanford.edu/data/glove.6B.zip, unzip, and place the file glove.6B.50d.txt in the project folder.
4.	Place the pre-trained model file lead_scoring_model.pth in the project root folder.

## Usage
1. Run the Streamlit web app
2. Open the local URL in your browser, upload your leads CSV file, and get scored results with deduplication.
3. CSV format should include columns:
    name, email, job_title, company, company_size, domain

## How It Works
1.	Job titles are converted into numeric vectors using pre-trained GloVe embeddings.
2.	Company size is normalized and combined with email validity features.
3.	The model predicts a lead quality score between 0 and 1.
4.	Leads are deduplicated based on email to avoid redundant contacts.
5.	Results are displayed and downloadable via the Streamlit app.

## Project Structure
1. indx.py: Streamlit frontend with lead scoring and deduplication logic
2. trainninference.ipynb: PyTorch lead scoring model and training code
3. glove.6B.50d.txt: GloVe embeddings file (not included, download separately)
4. lead_scoring_model.pth: Pre-trained PyTorch model weights
5. sample_leads.csv: Example leads CSV for testing
